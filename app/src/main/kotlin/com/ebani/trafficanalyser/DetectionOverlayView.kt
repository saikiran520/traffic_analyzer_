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

/**
 * Velocity-Aware Tracking Overlay.
 * Uses temporal projection to predict vehicle positions in realtime,
 * delivering a "nano-second" seamless live detection feel.
 */
class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        setShadowLayer(4f, 1f, 1f, Color.BLACK)
    }
    
    private val panelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xCC000000.toInt()
        style = Paint.Style.FILL
    }
    
    private class PredictiveBox(
        val label: String,
        val score: Float,
        var targetBox: RectF,
        var currentBox: RectF = RectF(targetBox)
    ) {
        var velocityX = 0f
        var velocityY = 0f
        var lastUpdateTime = System.currentTimeMillis()
        var isActive = true

        fun update(newBox: RectF) {
            val now = System.currentTimeMillis()
            val dt = max(1L, now - lastUpdateTime).toFloat()
            
            // Calculate current velocity (pixels per ms)
            velocityX = (newBox.centerX() - targetBox.centerX()) / dt
            velocityY = (newBox.centerY() - targetBox.centerY()) / dt
            
            targetBox.set(newBox)
            lastUpdateTime = now
            isActive = true
        }

        fun predict() {
            val now = System.currentTimeMillis()
            val dt = (now - lastUpdateTime).toFloat()
            
            // Project box forward based on velocity
            val dx = velocityX * dt
            val dy = velocityY * dt
            
            // Apply smoothing towards the projected target
            val lerpFactor = 0.25f
            currentBox.left = currentBox.left + (targetBox.left + dx - currentBox.left) * lerpFactor
            currentBox.top = currentBox.top + (targetBox.top + dy - currentBox.top) * lerpFactor
            currentBox.right = currentBox.right + (targetBox.right + dx - currentBox.right) * lerpFactor
            currentBox.bottom = currentBox.bottom + (targetBox.bottom + dy - currentBox.bottom) * lerpFactor
        }
    }
    
    private val trackedBoxes = mutableListOf<PredictiveBox>()
    private val counts = linkedMapOf<String, Int>()
    private val labelList = mutableListOf<String>()
    
    private var sourceWidth = 1
    private var sourceHeight = 1
    private var fps = 0f
    
    private val panelRect = RectF()
    private val drawBox = RectF()

    init {
        setWillNotDraw(false)
    }

    fun setLabels(labels: List<String>) {
        counts.clear()
        labelList.clear()
        labels.forEach { 
            val canonical = it.lowercase(Locale.US)
            counts[canonical] = 0 
            labelList.add(canonical)
        }
        invalidate()
    }

    fun update(newDetections: List<Detection>, width: Int, height: Int, newFps: Float) {
        synchronized(trackedBoxes) {
            sourceWidth = max(1, width)
            sourceHeight = max(1, height)
            fps = newFps

            labelList.forEach { counts[it] = 0 }
            
            // Mark all for cleanup
            trackedBoxes.forEach { it.isActive = false }

            newDetections.forEach { det ->
                val label = det.label.lowercase(Locale.US)
                counts[label] = (counts[label] ?: 0) + 1
                
                // Match with existing box using Centroid + IOU
                val existing = trackedBoxes.find { 
                    it.label == det.label && intersectionOverUnion(it.targetBox, det.box) > 0.2f 
                }
                
                if (existing != null) {
                    existing.update(det.box)
                } else {
                    trackedBoxes.add(PredictiveBox(det.label, det.score, RectF(det.box)))
                }
            }
            
            // Remove dead tracks
            trackedBoxes.removeAll { !it.isActive }
        }
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val scale = max(width / sourceWidth.toFloat(), height / sourceHeight.toFloat())
        val dx = (width - sourceWidth * scale) / 2f
        val dy = (height - sourceHeight * scale) / 2f

        synchronized(trackedBoxes) {
            trackedBoxes.forEach { det ->
                // APPLY PREDICTION ENGINE
                det.predict()

                val color = colorForLabel(det.label)
                boxPaint.color = color
                
                drawBox.set(
                    det.currentBox.left * scale + dx,
                    det.currentBox.top * scale + dy,
                    det.currentBox.right * scale + dx,
                    det.currentBox.bottom * scale + dy
                )
                
                canvas.drawRect(drawBox, boxPaint)
                textPaint.color = color
                canvas.drawText("${det.label}", drawBox.left, max(38f, drawBox.top - 10f), textPaint)
            }
        }

        drawSummary(canvas)
        
        // Continuous high-frequency redraw for predictive movement
        if (trackedBoxes.isNotEmpty()) {
            postInvalidateOnAnimation()
        }
    }

    private fun intersectionOverUnion(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun drawSummary(canvas: Canvas) {
        val visibleCount = min(12, labelList.size)
        panelRect.set(12f, 12f, 440f, 60f + visibleCount * 34f)
        canvas.drawRoundRect(panelRect, 8f, 8f, panelPaint)
        textPaint.color = Color.WHITE
        canvas.drawText("ULTRA-SYNC AI: ${fps.toInt()} Hz", 28f, 52f, textPaint)
        
        var r = 0
        labelList.forEach { label ->
            val c = counts[label] ?: 0
            if (c > 0 && r < visibleCount) {
                textPaint.color = colorForLabel(label)
                canvas.drawText("$label: $c", 28f, 90f + r * 34f, textPaint)
                r++
            }
        }
    }

    private fun colorForLabel(label: String): Int = when (label.lowercase(Locale.US)) {
        "hatchback", "sedan", "suv", "muv" -> Color.rgb(255, 180, 40)
        "bus", "truck", "lcv", "van" -> Color.rgb(40, 170, 255)
        "three-wheeler", "two-wheeler" -> Color.rgb(230, 90, 255)
        else -> Color.rgb(255, 255, 255)
    }
}
