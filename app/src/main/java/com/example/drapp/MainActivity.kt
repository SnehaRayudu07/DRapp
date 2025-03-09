package com.example.drapp

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.drapp.ml.ComFineTunedModelOptimized
//import com.example.drapp.ml.DrmulticlassificationFineTunedOptimized
import org.tensorflow.lite.DataType
//import org.tensorflow.lite.Interpreter
//import org.tensorflow.lite.support.model.Model
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
//import com.example.drapp.ml.FineTunedModel
//import com.example.drapp.ml.FineTunedModelKeras
//import com.example.drapp.ml.OptimizedModel

class MainActivity : AppCompatActivity() {

    private lateinit var camera: Button
    private lateinit var gallery: Button
    private lateinit var imageView: ImageView
    private lateinit var result: TextView
    private val imageSize = 512

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        camera = findViewById(R.id.button)
        gallery = findViewById(R.id.button2)
        result = findViewById(R.id.result)
        imageView = findViewById(R.id.imageView)

        camera.setOnClickListener {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, 3)
            } else {
                requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
            }
        }

        gallery.setOnClickListener {
            val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(galleryIntent, 1)
        }
    }

    private fun classifyImage(image: Bitmap) {
        try {
            val model = ComFineTunedModelOptimized.newInstance(applicationContext)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 512, 512, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            val intValues = IntArray(imageSize * imageSize)
            image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
            var pixel = 0

            for (i in 0 until imageSize) {
                for (j in 0 until imageSize) {
                    val value = intValues[pixel++]
                    byteBuffer.putFloat(((value shr 16) and 0xFF) * (1f / 255))
                    byteBuffer.putFloat(((value shr 8) and 0xFF) * (1f / 255))
                    byteBuffer.putFloat((value and 0xFF) * (1f / 255))
                }
            }
            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            val confidences = outputFeature0.floatArray
            val maxPos = confidences.indices.maxByOrNull { confidences[it] } ?: 0
            val classes = arrayOf("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")
            result.text = classes[maxPos]

            model.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            when (requestCode) {
                3 -> {
                    val image = data?.extras?.get("data") as? Bitmap
                    image?.let {
                        val dimension = minOf(it.width, it.height)
                        val thumbImage = ThumbnailUtils.extractThumbnail(it, dimension, dimension)
                        imageView.setImageBitmap(thumbImage)

                        val resizedImage = Bitmap.createScaledBitmap(thumbImage, imageSize, imageSize, false)
                        classifyImage(resizedImage)
                    }
                }
                1 -> {
                    val dat: Uri? = data?.data
                    try {
                        val image = MediaStore.Images.Media.getBitmap(this.contentResolver, dat)
                        imageView.setImageBitmap(image)
                        val resizedImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
                        classifyImage(resizedImage)
                    } catch (e: IOException) {
                        e.printStackTrace()
                    }
                }
            }
        }
    }
}
//package com.example.drapp
//
//import android.Manifest
//import android.content.Intent
//import android.content.pm.PackageManager
//import android.graphics.Bitmap
//import android.media.ThumbnailUtils
//import android.net.Uri
//import android.os.Bundle
//import android.provider.MediaStore
//import android.widget.Button
//import android.widget.ImageView
//import android.widget.TextView
////import androidx.activity.result.contract.ActivityResultContracts
////import androidx.annotation.Nullable
//import androidx.appcompat.app.AppCompatActivity
//import org.tensorflow.lite.DataType
//import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
//import java.io.IOException
//import java.nio.ByteBuffer
//import java.nio.ByteOrder
////import app.ij.mlwithtensorflowlite.ml.Model
//import com.example.drapp.ml.FineTunedModel
//import org.tensorflow.lite.Interpreter
//
////import org.tensorflow.lite.support.model.Model\
//
//class MainActivity : AppCompatActivity() {
//
//    private lateinit var camera: Button
//    private lateinit var gallery: Button
//    private lateinit var imageView: ImageView
//    private lateinit var result: TextView
//    private val imageSize = 512
//    val interpreter = Interpreter(tfliteModel, Interpreter.Options().setUseNNAPI(true))
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        setContentView(R.layout.activity_main)
//
//        camera = findViewById(R.id.button)
//        gallery = findViewById(R.id.button2)
//        result = findViewById(R.id.result)
//        imageView = findViewById(R.id.imageView)
//
//        camera.setOnClickListener {
//            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
//                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
//                startActivityForResult(cameraIntent, 3)
//            } else {
//                requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
//            }
//        }
//
//        gallery.setOnClickListener {
//            val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
//            startActivityForResult(galleryIntent, 1)
//        }
//    }
//
//    private fun classifyImage(image: Bitmap) {
//        try {
//
//            val model = FineTunedModel.newInstance(applicationContext)
//
//// Creates inputs for reference.
//            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 512, 512, 3), DataType.FLOAT32)
//            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
//            byteBuffer.order(ByteOrder.nativeOrder())
//
//            val intValues = IntArray(imageSize * imageSize)
//            image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
//            var pixel = 0
//
//            for (i in 0 until imageSize) {
//                for (j in 0 until imageSize) {
//                    val value = intValues[pixel++]
//                    byteBuffer.putFloat(((value shr 16) and 0xFF) * (1f / 1))
//                    byteBuffer.putFloat(((value shr 8) and 0xFF) * (1f / 1))
//                    byteBuffer.putFloat((value and 0xFF) * (1f / 1))
//                }
//            }
//            inputFeature0.loadBuffer(byteBuffer)
//
//            val outputs = model.process(inputFeature0)
//            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
//
//            val confidences = outputFeature0.floatArray
//            val maxPos = confidences.indices.maxByOrNull { confidences[it] } ?: 0
//            val classes = arrayOf("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")
//            result.text = classes[maxPos]
//
//            model.close()
//        } catch (e: IOException) {
//            e.printStackTrace()
//        }
//    }
//
//    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
//        super.onActivityResult(requestCode, resultCode, data)
//        if (resultCode == RESULT_OK) {
//            when (requestCode) {
//                3 -> {
//                    val image = data?.extras?.get("data") as? Bitmap
//                    image?.let {
//                        val dimension = minOf(it.width, it.height)
//                        val thumbImage = ThumbnailUtils.extractThumbnail(it, dimension, dimension)
//                        imageView.setImageBitmap(thumbImage)
//
//                        val resizedImage = Bitmap.createScaledBitmap(thumbImage, imageSize, imageSize, false)
//                        classifyImage(resizedImage)
//                    }
//                }
//                1 -> {
//                    val dat: Uri? = data?.data
//                    try {
//                        val image = MediaStore.Images.Media.getBitmap(this.contentResolver, dat)
//                        imageView.setImageBitmap(image)
//                        val resizedImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
//                        classifyImage(resizedImage)
//                    } catch (e: IOException) {
//                        e.printStackTrace()
//                    }
//                }
//            }
//        }
//    }
//}
