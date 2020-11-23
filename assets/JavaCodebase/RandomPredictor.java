// Import to import GatewayServer
import py4j.GatewayServer;

import java.util.Scanner;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.FileNotFoundException;
import java.io.File;
import java.util.Random;

public class RandomPredictor {
    private int seed;
    private String[] vocabulary;

    public void setSeed(int value) {
        seed = value;
    }

    public void readVocabulary(String filename) throws FileNotFoundException {
        System.out.println("Reading file from : " + filename);
        System.out.println("Current working directory : " + System.getProperty("user.dir"));
        Scanner sc = new Scanner(new File(filename));
        List<String> lines = new ArrayList<String>();
        while (sc.hasNextLine()) {
            lines.add(sc.nextLine());
        }
        vocabulary = lines.toArray(new String[0]);
        System.out.println("Vocabulary loaded in memory : ");
        System.out.println(Arrays.toString(vocabulary));
    }

    public List<List<String>> predict(String smile) {
        List<List<String>> sentences = new ArrayList<List<String>>();
        for (int i = 0; i < 3; i++) {
            List<String> words = new ArrayList<String>();
            for (int j = 0; j < 5; j++) {
                int rnd = new Random().nextInt(vocabulary.length);
                words.add(vocabulary[rnd]);
            }
            sentences.add(words);
        }
        return sentences;
    }

    public static void main(String[] args) {
        RandomPredictor app = new RandomPredictor();
        // app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(app);
        server.start();
    }
}
