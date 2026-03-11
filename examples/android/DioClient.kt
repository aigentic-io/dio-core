/**
 * DIO Android Client — reference implementation for the NYC Android Meetup talk.
 *
 * Shows how an Android app calls the DIO REST sidecar using Retrofit + OkHttp + Coroutines.
 * ClientContext captures real-time device state (battery, connectivity, on-device model).
 * User identity (tier, org, policies) is resolved server-side from the Bearer token —
 * never self-reported by the client.
 *
 * Request headers sent on every call:
 *   Authorization: Bearer <token>      — JWT; server resolves user_id, tier, policies
 *   Accept-Language: <device locale>   — ordered language preference (RFC 7231)
 *   X-Session-ID: <uuid>              — stable per conversation, reset on new chat
 *   X-Request-ID: <uuid>              — unique per request; echoed in response for tracing
 *
 * Setup:
 *   1. Deploy DIO server: uvicorn aigentic.server.app:app  (Fly.io, homelab, etc.)
 *   2. Add to build.gradle.kts:
 *        implementation("com.squareup.retrofit2:retrofit:2.9.0")
 *        implementation("com.squareup.retrofit2:converter-gson:2.9.0")
 *        implementation("com.squareup.okhttp3:okhttp:4.12.0")
 *        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
 *   3. Set DIO_BASE_URL in BuildConfig or local.properties
 */

package io.aigentic.dio.android

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import com.google.gson.annotations.SerializedName
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import java.util.UUID

// ── Data classes ──────────────────────────────────────────────────────────────

/**
 * Device/environment state. Changes per request as battery drains and
 * network switches. Works the same on iOS (via Swift client) and web.
 *
 * Note: user identity (tier, org, policies) is NOT included here.
 * The server resolves all user context from the Authorization Bearer token.
 */
data class ClientContext(
    @SerializedName("platform")         val platform: String = "android",   // "android" | "ios" | "web" | "desktop" | "server"
    @SerializedName("connectivity")     val connectivity: String,            // "wifi" | "cellular" | "ethernet" | "offline"
    @SerializedName("battery_level")    val batteryLevel: Int?,              // 0-100; null = plugged / not applicable
    @SerializedName("on_device_model")  val onDeviceModel: String? = null,   // "gemini-nano", "phi-3-mini", null
    @SerializedName("memory_mb")        val memoryMb: Int? = null,           // available RAM hint for on-device feasibility
)

/**
 * A single turn in the conversation. Follows the OpenAI / OpenRouter / LiteLLM
 * message format so you can reuse the same message construction code across providers.
 * `content` is a String for plain text; for multimodal pass a List<ContentPart>.
 */
data class Message(
    @SerializedName("role")    val role: String,  // "system" | "user" | "assistant"
    @SerializedName("content") val content: Any,  // String or List<ContentPart>
)

/** Text content part for multimodal messages. */
data class TextPart(
    @SerializedName("type") val type: String = "text",
    @SerializedName("text") val text: String,
)

/** Image content part for multimodal messages. */
data class ImagePart(
    @SerializedName("type")      val type: String = "image_url",
    @SerializedName("image_url") val imageUrl: ImageUrl,
)

data class ImageUrl(
    @SerializedName("url")    val url: String,             // HTTPS URL or data:image/...;base64,...
    @SerializedName("detail") val detail: String = "auto", // "auto" | "low" | "high"
)

/**
 * Request to infer. Uses the OpenAI / OpenRouter / LiteLLM `messages` format.
 *
 * Standard inference params (temperature, maxTokens) are passed through to the
 * selected provider. DIO-specific FDE overrides are additive on top.
 */
data class InferRequest(
    @SerializedName("messages")         val messages: List<Message>,
    @SerializedName("client_context")   val clientContext: ClientContext,
    @SerializedName("temperature")      val temperature: Double? = null,
    @SerializedName("max_tokens")       val maxTokens: Int? = null,
    @SerializedName("max_cost")         val maxCost: Double? = null,
    @SerializedName("max_latency_ms")   val maxLatencyMs: Int? = null,
    @SerializedName("require_local")    val requireLocal: Boolean? = null,
)

data class InferResult(
    @SerializedName("provider")    val provider: String,
    @SerializedName("model")       val model: String?,
    @SerializedName("content")     val content: String,
    @SerializedName("routed_by")   val routedBy: String,
    @SerializedName("metadata")    val metadata: Map<String, Any>,
)

data class ProviderInfo(
    val name: String,
    val type: String,
    val model: String?,
    val capability: Double,
)

// ── Retrofit service interface ─────────────────────────────────────────────────

interface DioApi {
    @GET("health")
    suspend fun health(): Map<String, Any>

    @GET("providers")
    suspend fun providers(): List<ProviderInfo>

    @POST("infer")
    suspend fun infer(@Body request: InferRequest): InferResult
}

// ── ClientContext collector ────────────────────────────────────────────────────

object ClientContextCollector {
    /**
     * Collects real Android device state. Call before each request so routing
     * reflects current conditions (battery may drain, network may switch).
     */
    fun current(context: Context): ClientContext {
        return ClientContext(
            platform = "android",
            connectivity = getConnectivity(context),
            batteryLevel = getBatteryLevel(context),
            onDeviceModel = detectOnDeviceModel(),
            memoryMb = getAvailableMemoryMb(context),
        )
    }

    private fun getBatteryLevel(context: Context): Int? {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val level = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        return if (level == Integer.MIN_VALUE) null else level
    }

    private fun getConnectivity(context: Context): String {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = cm.activeNetwork ?: return "offline"
        val caps = cm.getNetworkCapabilities(network) ?: return "offline"
        return when {
            caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)     -> "wifi"
            caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "ethernet"
            caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "cellular"
            else -> "offline"
        }
    }

    private fun detectOnDeviceModel(): String? {
        // Gemini Nano is available on Pixel 9+ via Android AICore API.
        // Check Build.MODEL or query AICore availability in production.
        return null  // replace with real AICore availability check
    }

    private fun getAvailableMemoryMb(context: Context): Int? {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val info = android.app.ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        return (info.availMem / (1024 * 1024)).toInt()
    }
}

// ── DioClient singleton ────────────────────────────────────────────────────────

object DioClient {
    private const val BASE_URL = "https://your-dio-server.fly.dev/"  // replace with your deployment

    // Stable for the lifetime of a conversation. Reset when the user starts a new chat.
    var sessionId: String = UUID.randomUUID().toString()
        private set

    fun resetSession() {
        sessionId = UUID.randomUUID().toString()
    }

    /**
     * Build the Retrofit API client. Call once per auth token (re-build on token refresh).
     *
     * Headers injected on every request via OkHttp interceptor:
     *   Authorization  — Bearer token from your auth provider (Auth0, Firebase, Cognito, etc.)
     *   Accept-Language — device locale list as RFC 7231 ordered preference
     *   X-Session-ID   — stable per conversation; server groups log lines by session
     *   X-Request-ID   — unique per request; echoed back in response header for tracing
     */
    fun build(authToken: String): DioApi {
        val client = OkHttpClient.Builder()
            .addInterceptor(Interceptor { chain ->
                val req = chain.request().newBuilder()
                    .header("Authorization", "Bearer $authToken")
                    .header("Accept-Language", buildAcceptLanguage())
                    .header("X-Session-ID", sessionId)
                    .header("X-Request-ID", UUID.randomUUID().toString())
                    .build()
                chain.proceed(req)
            })
            .build()

        return Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(DioApi::class.java)
    }

    /**
     * Build an RFC 7231 Accept-Language value from the device locale list.
     * Example output: "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7"
     * The server logs this and will use it for language-aware model selection (future).
     */
    private fun buildAcceptLanguage(): String {
        val locales = androidx.core.os.LocaleListCompat.getAdjustedDefault()
        val tags = (0 until locales.size()).mapNotNull { locales[it]?.toLanguageTag() }
        return tags.mapIndexed { i, tag ->
            if (i == 0) tag else "$tag;q=${String.format("%.1f", maxOf(0.1, 1.0 - i * 0.1))}"
        }.joinToString(",")
    }
}

// ── Usage example ─────────────────────────────────────────────────────────────
//
// class ChatViewModel(
//     private val context: Context,
//     private val authToken: String,   // from your auth provider (Auth0, Firebase, Cognito, etc.)
// ) : ViewModel() {
//
//     private val api = DioClient.build(authToken)
//
//     fun sendMessage(text: String) {
//         viewModelScope.launch {
//             val result = api.infer(
//                 InferRequest(
//                     messages = listOf(Message(role = "user", content = text)),
//                     clientContext = ClientContextCollector.current(context),
//                 )
//             )
//             // DIO routes automatically — server resolves user tier/policies from token:
//             // - Low battery       → on-device (no HTTP drain)
//             // - Offline           → on-device only
//             // - Free tier (token) → cost cap blocks frontier models
//             // - Premium (token)   → frontier cloud for complex reasoning
//             // - PII detected      → on-device (never sent to cloud)
//             _uiState.value = result.content
//
//             // Check which provider was chosen and why (useful for debugging):
//             val requestId = result.metadata["request_id"]  // echoed from X-Request-ID header
//         }
//     }
//
//     fun startNewConversation() {
//         DioClient.resetSession()  // new X-Session-ID for the next conversation
//     }
// }
