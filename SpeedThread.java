/*
speed test
qty 2 prefilled array size [12k, 12k] multiplying into the first,
10 iterations
highly variable results: 650 700 773 818 820 853 920 922 1100
20 iterations: 2140 2260 2290 2340 2370
*/

import java.util.Arrays;
import java.util.Random;
public class SpeedThread {
    public static final int xSize = 12000;
    public static final int ySize = 12000;
    public static final int iterations = 20;
    static double[][] aArray = new double[ySize][xSize];
    double[][] bArray = new double[ySize][xSize];
    
    // constructor
    public SpeedThread(){
        for(int y = 0; y < ySize; y++) {
            for(int x = 0; x < xSize; x++){
                // prefill with random doubles
                aArray[y][x] = (new Random().nextDouble() * 2);
                bArray[y][x] = (new Random().nextDouble() * 2);
            }
        }
    }
    
    static class Process implements Runnable {
        private double[][] bArr;

        // instance constructor
        public Process(double[][] aArr, double[][] bArr) {
            this.bArr = bArr;
        }

        @Override
        public void run() {
            // multiply
            for(int y = 0; y < ySize; y++) {
                for(int x = 0; x < xSize; x++){
                    // multiply both into aArray
                    aArray[y][x] *= this.bArr[y][x];
                }
            }
        }
    }
    
    public void postAdd() {
        for(int y = 0; y < ySize; y++) {
            for(int x = 0; x < xSize; x++){
                aArray[y][x] += 1.2;
                bArray[y][x] += 1.2;
            }
        }
    }
    
    public void check() {
        // only check when x and ySizes are small, < 100
        System.out.println("a");
        System.out.println(Arrays.deepToString(aArray));
        System.out.println("b");
        System.out.println(Arrays.deepToString(bArray));
    }
    
    public static void main(String[] args) {
        System.out.println("done compiling");
        // constructor contains prefilling with random doubles .0001 - 2.000
        SpeedThread thisTest = new SpeedThread();
        
        // create array of threads
        Thread myThreads[] = new Thread[iterations];
        
        // start timer
        long startTime = System.currentTimeMillis();
        System.out.println("started");
        
        // process threads
        for(int i = 0; i < iterations; i++){
            myThreads[i] = new Thread(new Process(aArray, thisTest.bArray));
            myThreads[i].start();
        }
        // wait for all to complete
        for(int i = 0; i < iterations; i++){
            try {
                myThreads[i].join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        
        // stop timer
        long endTime = System.currentTimeMillis();
        long duration = (endTime - startTime) / 1;
        System.out.println(duration + "ms");
        
        //thisTest.check();
        System.out.println(thisTest.bArray[0][0]);
        
        thisTest.postAdd();
    }
}