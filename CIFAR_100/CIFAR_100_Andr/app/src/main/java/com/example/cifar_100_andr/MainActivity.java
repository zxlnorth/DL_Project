package com.example.cifar_100_andr;

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

    private static final String MODEL_FILE = "file:///android_asset/cifar100.pb";
    private static final String INPUT_NODE = "reshape_1_input";
    private static final long[] INPUT_SHAPE = {1, 3072};
    private static final String OUTPUT_NODE = "dense_2/Softmax";
    private TensorFlowInferenceInterface inferenceInterface;

    int imageIDsIndex = 9;
    int[] imageIDs = {
            R.drawable.apples,
            R.drawable.baby,
            R.drawable.beetle,
            R.drawable.bicycle,
            R.drawable.butterfly,
            R.drawable.forest,
            R.drawable.rocket,
            R.drawable.skunk,
            R.drawable.tulip,
            R.drawable.whale
    };
    Bitmap chosenImageBitmap;
    String[] labels = {" apples", " aquarium fish", " baby", " bear", " beaver", " bed", " bee", " beetle", " bicycle",
            " bottles", " bowls", " boy", " bridge", " bus", " butterfly", " camel", " cans", " castle", " caterpillar",
            " cattle", " chair", " chimpanzee", " clock", " cloud", " cockroach", " computer keyboard", " couch",
            " crab", " crocodile", " cups", " dinosaur", " dolphin", " elephant", " flatfish", " forest", " fox",
            " girl", " hamster", " house", " kangaroo", " lamp", " lawn-mower", " leopard", " lion", " lizard",
            " lobster", " man", " maple", " motorcycle", " mountain", " mouse", " mushrooms", " oak", " oranges",
            " orchids", " otter", " palm", " pears", " pickup truck", " pine", " plain", " plates", " poppies",
            " porcupine", " possum", " rabbit", " raccoon", " ray", " road", " rocket", " roses", " sea", " seal",
            " shark", " shrew", " skunk", " skyscraper", " snail", " snake", " spider", " squirrel", " streetcar",
            " sunflowers", " sweet peppers", " table", " tank", " telephone", " television", " tiger", " tractor",
            " train", " trout", " tulips", " turtle", " wardrobe", " whale", " willow", " wolf", " woman", " worm"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = (ImageView) findViewById(R.id.image_view);
        textView = (TextView) findViewById(R.id.text_view);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    }

    public void loadImageAction(View view) {
        imageIDsIndex = (imageIDsIndex >= 9) ? 0 : imageIDsIndex + 1;
        chosenImageBitmap = BitmapFactory.decodeResource(getResources(), imageIDs[imageIDsIndex]);
        chosenImageBitmap = Bitmap.createScaledBitmap(chosenImageBitmap, 32, 32, true);
        imageView.setImageBitmap(chosenImageBitmap);
    }

    public void predictImageAction(View view) {
        float[] pixelBuffer = convertImageToFloatArray();
        float[] results = runInference(pixelBuffer);
        displayResults(results);
    }

    private float[] convertImageToFloatArray() {
        int[] pixels = new int[1024];
        chosenImageBitmap.getPixels(pixels, 0, 32, 0, 0, 32, 32);
        float[] floatArray = new float[3072];
        for (int i = 0; i < 1024; i++) {
            floatArray[i] = ((pixels[i] >> 16) & 0xff) / 255.0f;
            floatArray[i + 1] = ((pixels[i] >> 8) & 0xff) / 255.0f;
            floatArray[i + 2] = (pixels[i] & 0xff) / 255.0f;
        }
        return floatArray;
    }

    private float[] runInference(float[] pixelBuffer) {
        inferenceInterface.feed(INPUT_NODE, pixelBuffer, INPUT_SHAPE);
        inferenceInterface.run(new String[] {OUTPUT_NODE});
        float[] results = new float[100];
        inferenceInterface.fetch(OUTPUT_NODE, results);
        return results;
    }

    private void displayResults(float[] results) {
        float max = results[0];
        int maxIndex = 0;
        for (int i = 0; i < results.length; i++) {
            if (results[i] > max) {
                max = results[i];
                maxIndex = i;
            }
        }
        textView.setText("Model predicts:" + labels[maxIndex]);
    }
}
