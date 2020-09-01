package com.example.emotions_detector_andr;

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
    TextView textView;

    static {
        System.loadLibrary("tensorflow_inference");
    }
    private static final String MODEL_FILE = "file:///android_asset/emotions_detector.pb";
    private static final String INPUT_NODE = "reshape_1_input";
    private static final long[] INPUT_SHAPE = {1, 3072};
    private static final String OUTOUT_NODE = "dense_2/Softmax";
    private TensorFlowInferenceInterface inferenceInterface;

    int imageIDsIndex = 9;
    int[] imageIDs = {
            R.drawable.happy0,
            R.drawable.happy1,
            R.drawable.happy2,
            R.drawable.happy3,
            R.drawable.happy4,
            R.drawable.sad0,
            R.drawable.sad1,
            R.drawable.sad2,
            R.drawable.sad3,
            R.drawable.sad4
    };
    Bitmap displayImageBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = (ImageView) findViewById(R.id.image_view);
        textView = (TextView) findViewById(R.id.results_text_view);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    }

    public void loadImageAction(View view) {
        imageIDsIndex = (imageIDsIndex >= 9) ? 0 : imageIDsIndex + 1;
        Bitmap imageBitmap = BitmapFactory.decodeResource(getResources(), imageIDs[imageIDsIndex]);
        displayImageBitmap = Bitmap.createScaledBitmap(imageBitmap, 32, 32, true);
        imageView.setImageBitmap(displayImageBitmap);
    }

    public void guessImageAction(View view) {
        float[] pixelBuffer = convertImageToFloatArray();
        float[] results = performInference(pixelBuffer);
        displayResults(results);
    }

    private float[] convertImageToFloatArray() {
        int[] intArray = new int[2014];
        displayImageBitmap.getPixels(intArray, 0, 32, 0, 0, 32, 32);
        float[] floatArray = new float[3072];
        for (int i = 0; i < 1024; i++) {
            floatArray[i] = ((intArray[i] >> 16) & 0xff) / 255.0f;
            floatArray[i + 1] = ((intArray[i] >> 8) & 0xff) / 255.0f;
            floatArray[i + 2] = (intArray[i] & 0xff) / 255.0f;
        }
        return floatArray;
    }

    private float[] performInference(float[] pixelBuffer) {
        inferenceInterface.feed(INPUT_NODE, pixelBuffer, INPUT_SHAPE);
        inferenceInterface.run(new String[] {OUTOUT_NODE});
        float[] results = new float[2];
        inferenceInterface.fetch(OUTOUT_NODE, results);
        return results;
    }

    private void displayResults(float[] results) {
        if (results[0] >= results[1])
            textView.setText("Model predicts: Happy");
        else if (results[0] < results[1])
            textView.setText("Model predicts: Sad");
        else textView.setText("Model predicts: Neither");
    }
}
