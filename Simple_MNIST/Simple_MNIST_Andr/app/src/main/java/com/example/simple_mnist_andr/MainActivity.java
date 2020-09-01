package com.example.simple_mnist_andr;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {
    // UI elements
    ImageView imageView;
    TextView textView;

    // variables for communicating with the model file
    static {
        System.loadLibrary("tensorflow_inference");
    }
    private static final String MODEL_FILE = "file:///android_asset/optimized_frozen_mnist_model.pb";
    private static final String INPUT_NODE = "x_input";
    private static final int[] INPUT_SHAPE = {1, 784};  // import one image at a time
    private static final String OUTPUT_NODE = "y_actual";
    private TensorFlowInferenceInterface inferenceInterface;

    // variables to help hold the images in the drawable folder and iterate through the list
    private int imageListIndex = 9;
    private final int[] imageIdList = {
            R.drawable.digit0,
            R.drawable.digit1,
            R.drawable.digit2,
            R.drawable.digit3,
            R.drawable.digit4,
            R.drawable.digit5,
            R.drawable.digit6,
            R.drawable.digit7,
            R.drawable.digit8,
            R.drawable.digit9,
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // set up the UI elements
        imageView = (ImageView) findViewById(R.id.image_view);
        textView = (TextView) findViewById(R.id.text_view);

        // initialize the inference variable to use the model
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
    }

    public void predictDigitClick(View view) {
        // get the image data as a float array
        float[] pixelBuffer = convertImage();
        // get the label that represents the prediction
        float[] results = formPrediction(pixelBuffer);
//        for(float result : results) {
//            Log.d("results", String.valueOf(result));
//        }
        printResults(results);
    }

    private void printResults(float[] results) {
        float max = 0;
        float secondMax = 0;
        int maxIndex = 0;
        int secondMaxIndex = 0;
        for (int i = 0; i < 10; i++) {
            if (results[i] > max) {
                secondMax = max;
                secondMaxIndex = maxIndex;
                max = results[i];
                maxIndex = i;
            } else if (results[i] < max && results[i] > secondMax) {
                secondMax = results[i];
                secondMaxIndex = i;
            }
        }
        String output = "Model predicts: " + String.valueOf(maxIndex) +
                ",  second choice: " + String.valueOf(secondMaxIndex);
        textView.setText(output);
    }

    /**
     * function to actually make the prediction
     * takes in array of floats that represents the image data
     * outputs an array of floats that represents the label based on the current prediction
     */
    private float[] formPrediction(float[] pixelBuffer) {
        // fill input node with the pixel buffer
        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SHAPE, pixelBuffer);
        // make prediction by running inference on the model and store results in output node
        inferenceInterface.runInference(new String[] {OUTPUT_NODE});
        float[] results = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        inferenceInterface.readNodeFloat(OUTPUT_NODE, results);
        return results;
    }

    // convert currently displayed image into a float array to feed into the model
    private float[] convertImage() {
        // convert current image to a scaled 28*28 bitmap
        Bitmap imageBitmap = BitmapFactory.decodeResource(getResources(),
                imageIdList[imageListIndex]);
        imageBitmap = Bitmap.createScaledBitmap(imageBitmap, 28, 28, true);
        imageView.setImageBitmap(imageBitmap);
        int[] imageAsIntArray = new int[784];
        float[] imageAsFloatArray = new float[784];
        // get the pixel values of the bitmap and store them in a flattened in array
        imageBitmap.getPixels(imageAsIntArray, 0, 28, 0, 0, 28, 28);
        for (int i = 0; i < 784; i++) {
            imageAsFloatArray[i] = imageAsIntArray[i] / -16777216;  // not exactly between 0-1
        }
        return imageAsFloatArray;
    }

    public void loadNextImageClick(View view) {
        if (imageListIndex >= 9) {
            imageListIndex = 0;
        } else {
            imageListIndex += 1;
        }
        // imageListIndex = (imageListIndex >= 9) ? 0 : imageListIndex + 1;
        imageView.setImageDrawable(getDrawable(imageIdList[imageListIndex]));
    }
}
