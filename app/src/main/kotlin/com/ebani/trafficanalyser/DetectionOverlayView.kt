package com.ebani.trafficanalyser

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        setShadowLayer(4f, 1f, 1f, Color.BLACK)
    }
    private val panelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xAA000000.toInt()
        style = Paint.Style.FILL
    }
    private val detections = mutableListOf<Detection>()
    private val counts = linkedMapOf<String, Int>()
    private var sourceWidth = 1
    private var sourceHeight = 1
    private var fps = 0f

    init {
        setWillNotDraw(false)
    }

    fun setLabels(labels: List<String>) {
        counts.clear()
        labels.forEach { counts[it.canonical()] = 0 }
        invalidate()
    }

    fun update(newDetections: List<Detection>, width: Int, height: Int, newFps: Float) {
        detections.clear()
        detections.addAll(newDetections)
        sourceWidth = max(1, width)
        sourceHeight = max(1, height)
        fps = newFps

        counts.keys.toList().forEach { counts[it] = 0 }
        detections.forEach { detection ->
            val label = detection.label.canonical()
            counts[label] = (counts[label] ?: 0) + 1
        }
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scale = max(width / sourceWidth.toFloat(), height / sourceHeight.toFloat())
        val dx = (width - sourceWidth * scale) / 2f
        val dy = (height - sourceHeight * scale) / 2f

        detections.forEach { detection ->
            val color = colorForLabel(detection.label)
            boxPaint.color = color
            val box = detection.box.map(scale, dx, dy)
            canvas.drawRect(box, boxPaint)
            textPaint.color = color
            canvas.drawText(
                String.format(Locale.US, "%s %.2f", detection.label, detection.score),
                box.left,
                max(38f, box.top - 10f),
                textPaint,
            )
        }

        drawCounts(canvas)
    }

    private fun RectF.map(scale: Float, dx: Float, dy: Float) = RectF(
        left * scale + dx,
        top * scale + dy,
        right * scale + dx,
        bottom * scale + dy,
    )

    private fun drawCounts(canvas: Canvas) {
        val visible = min(14, counts.size)
        val rowHeight = 34f
        val panelWidth = min(width - 24f, 460f)
        val panelHeight = 58f + visible * rowHeight
        canvas.drawRoundRect(RectF(12f, 12f, 12f + panelWidth, 12f + panelHeight), 8f, 8f, panelPaint)

        textPaint.color = Color.WHITE
        canvas.drawText(String.format(Locale.US, "FPS %.1f", fps), 28f, 52f, textPaint)

        counts.entries.take(visible).forEachIndexed { row, entry ->
            textPaint.color = colorForLabel(entry.key)
            canvas.drawText("${entry.key}: ${entry.value}", 28f, 90f + row * rowHeight, textPaint)
        }
    }

    private fun String.canonical() = lowercase(Locale.US)

    private fun colorForLabel(label: String): Int = when (label.canonical()) {
        "hatchback" -> Color.rgb(255, 180, 40)
        "sedan" -> Color.rgb(255, 210, 60)
        "suv" -> Color.rgb(255, 140, 80)
        "muv" -> Color.rgb(255, 120, 120)
        "bus" -> Color.rgb(40, 170, 255)
        "truck" -> Color.rgb(80, 230, 130)
        "three-wheeler" -> Color.rgb(230, 90, 255)
        "two-wheeler" -> Color.rgb(255, 80, 180)
        "lcv" -> Color.rgb(90, 255, 170)
        "mini-bus" -> Color.rgb(80, 200, 255)
        "tempo-traveller" -> Color.rgb(130, 220, 255)
        "bicycle" -> Color.rgb(180, 255, 80)
        "van" -> Color.rgb(120, 210, 255)
        else -> {
            val seed = abs(label.hashCode())
            Color.rgb(80 + seed % 176, 80 + (seed / 7) % 176, 80 + (seed / 17) % 176)
        }
    }
}
