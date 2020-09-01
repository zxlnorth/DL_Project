package com.example.face_recognition_andr;

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
    private static final String MODEL_FILE = "file:///android_asset/face_recognition.pb";
    private static final String INPUT_NODE = "reshape_1_input";
    private static long[] INPUT_SHAPE = {1, 3072};
    private static final String OUTPUT_NODE = "dense_2/Softmax";
    private TensorFlowInferenceInterface inferenceInterface;

    int imageIDsIndex = 19;
    int[] imageIDs = {
            R.drawable.face0,
            R.drawable.aiplane,
            R.drawable.face1,
            R.drawable.automobile,
            R.drawable.face2,
            R.drawable.bird,
            R.drawable.face3,
            R.drawable.cat,
            R.drawable.face4,
            R.drawable.deer,
            R.drawable.face5,
            R.drawable.dog,
            R.drawable.face6,
            R.drawable.frog,
            R.drawable.face7,
            R.drawable.horse,
            R.drawable.face8,
            R.drawable.ship,
            R.drawable.face9,
            R.drawable.truck
    };
    Bitmap displayItemBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = (ImageView) findViewById(R.id.image_view);
        textView = (TextView) findViewById(R.id.results_text_view);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    }

    public void loadNextImageAction(View view) {
        imageIDsIndex = (imageIDsIndex >= 19) ? 0 : imageIDsIndex + 1;
        //imageView.setImageResource(imageIDs[imageIDsIndex]);
        displayItemBitmap = BitmapFactory.decodeResource(getResources(), imageIDs[imageIDsIndex]);
        displayItemBitmap = Bitmap.createScaledBitmap(displayItemBitmap, 32, 32, true);
        imageView.setImageBitmap(displayItemBitmap);
    }

    public void predictImageAction(View view) {
        float[] pixelBuffer = convertImageToFloats();
        float[] results = makePrediction(pixelBuffer);
        displayResults(results);
    }

    private float[] convertImageToFloats() {
        int[] intArray = new int[1024];
        displayItemBitmap.getPixels(intArray, 0, 32, 0, 0, 32, 32);
        float[] floatArray = new float[3072];
        for (int i = 0; i < 1024; i++) {
            floatArray[i] = ((intArray[i] >> 16) & 0xff) / 255.0f;  // red
            floatArray[i + 1] = ((intArray[i] >> 8) & 0xff) / 255.0f;  // green
            floatArray[i + 2] = (intArray[i] & 0xff) / 255.0f;  // blue
        }
        return floatArray;
    }

    private float[] makePrediction(float[] pixelBuffer) {
        inferenceInterface.feed(INPUT_NODE, pixelBuffer, INPUT_SHAPE);
        inferenceInterface.run(new String[] {OUTPUT_NODE});
        float[] results = new float[2];
        inferenceInterface.fetch(OUTPUT_NODE, results);
        return results;
    }

    private void displayResults(float[] results) {
        if (results[0] > results[1])
            textView.setText("Prediction: Face");
        else if (results[1] > results[0])
            textView.setText("Prediction: Not a face");
        else textView.setText("Prediction: Not sure");
    }
}
