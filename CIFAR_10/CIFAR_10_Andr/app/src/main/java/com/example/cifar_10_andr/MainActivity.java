package com.example.cifar_10_andr;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    TextView resultsTextView;

    static {
        System.loadLibrary("tensorflow_inference");
    }
    private static final String MODEL_FILE = "file:///android_asset/cifar10.pb";
    private static final String INPUT_NODE = "reshape_1_input";
    private static final long[] INPUT_SHAPE = {1, 3072};
    private static final String OUTPUT_NODE = "dense_2/Softmax";
    TensorFlowInferenceInterface inferenceInterface;

    int imageArrayIndex = 9;
    int[] imageResourceIds = {
            R.drawable.aiplane,
            R.drawable.automobile,
            R.drawable.bird,
            R.drawable.cat,
            R.drawable.deer,
            R.drawable.dog,
            R.drawable.frog,
            R.drawable.horse,
            R.drawable.ship,
            R.drawable.truck,
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = (ImageView) findViewById(R.id.image_view);
        resultsTextView = (TextView) findViewById(R.id.results_text_view);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    }

    public void nextImageAction(View view) {
        imageArrayIndex = (imageArrayIndex >= 9) ? 0 : imageArrayIndex + 1;
        imageView.setImageResource(imageResourceIds[imageArrayIndex]);
    }

    public void guessImageAction(View view) {
        float[] pixelBuffer = formatImageData();
        float[] results = makePrediction(pixelBuffer);
        displayResults(results);
    }

    private float[] formatImageData() {
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), imageResourceIds[imageArrayIndex]);
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 32, 32, true);
        int[] intArray = new int[1024];
        scaledBitmap.getPixels(intArray, 0, 32, 0, 0, 32, 32);
        float[] floatArray = new float[3072];
        for (int i = 0; i < 1024; i++) {
            floatArray[i] = (intArray[i] >> 16 & 0xff) / 255.0f;
            floatArray[i + 1] = (intArray[i] >> 8 & 0xff) / 255.0f;
            floatArray[i + 2] = (intArray[i] & 0xff) / 255.0f;
        }
        return floatArray;
    }

    private float[] makePrediction(float[] pixelBuffer) {
        inferenceInterface.feed(INPUT_NODE, pixelBuffer, INPUT_SHAPE);
        inferenceInterface.run(new String[] {OUTPUT_NODE});
        float[] results = new float[10];
        inferenceInterface.fetch(OUTPUT_NODE, results);
        return results;
    }

    private void displayResults(float[] resultsArray) {
        String[] answers = {
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck"
        };
        int maxIndex = 0;
        float max = 0;
        int secondMaxIndex = 0;
        float secondMax = 0;
        for (int i = 0; i < 10; i++) {
            if (resultsArray[i] > max) {
                secondMax = max;
                secondMaxIndex = maxIndex;
                max = resultsArray[i];
                maxIndex = i;
            } else if (resultsArray[i] < max && resultsArray[i] > secondMax) {
                secondMax = resultsArray[i];
                secondMaxIndex = i;
            }
        }
        String answer = answers[maxIndex];
        String secondAnswer = answers[secondMaxIndex];
        resultsTextView.setText("Prediction: " + answer + "\nSecond prediction: " + secondAnswer);
    }
}
