package com.example.advanced_mnist_andr;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    TextView resultsTextView;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE = "file:///android_asset/optimized_advanced_mnist.pb";
    private static final String INPUT_NODE = "x_input";
    private static final int[] INPUT_SIZE = {1, 784};
    private static final String KEEP_PROB = "keep_prob";
    private static final int[] KEEP_PROB_SIZE = {1};
    private static final String OUTPUT_NODE = "y_readout1";
    private TensorFlowInferenceInterface inferenceInterface;

    private int imageIndex = 9;
    private int[] imageResourceIDs = {
            R.drawable.digit0,
            R.drawable.digit1,
            R.drawable.digit2,
            R.drawable.digit3,
            R.drawable.digit4,
            R.drawable.digit5,
            R.drawable.digit6,
            R.drawable.digit7,
            R.drawable.digit8,
            R.drawable.digit9
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = (ImageView) findViewById(R.id.image_view);
        resultsTextView = (TextView) findViewById(R.id.results_text_view);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
    }

    public void loadImageAction(View view) {
        imageIndex = (imageIndex >= 9) ? 0 : imageIndex + 1;
        imageView.setImageResource(imageResourceIDs[imageIndex]);
    }

    public void predictDigitAction(View view) {
        float[] pixelBuffer = convertImage();
        float[] results = predictDigit(pixelBuffer);
        formatResults(results);
    }

    private void formatResults(float[] results) {
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
                ", second choice: " + String.valueOf(secondMaxIndex);
        resultsTextView.setText(output);
    }

    private float[] predictDigit(float[] pixelBuffer) {
        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, pixelBuffer);
        inferenceInterface.fillNodeFloat(KEEP_PROB, KEEP_PROB_SIZE, new float[] {1.0f});
        inferenceInterface.runInference(new String[] {OUTPUT_NODE});
        float[] outputs = new float[10];
        inferenceInterface.readNodeFloat(OUTPUT_NODE, outputs);
        return outputs;
    }

    private float[] convertImage() {
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), imageResourceIDs[imageIndex]);
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true);
        imageView.setImageBitmap(scaledBitmap);
        int[] intArray = new int[784];
        float[] floatArray = new float[784];
        scaledBitmap.getPixels(intArray, 0, 28, 0, 0, 28, 28);
        for (int i = 0; i < 784; i++) {
            floatArray[i] = intArray[i] / -16777216;
        }
        return floatArray;
    }
}
