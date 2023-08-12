/*
speed test
qty 2 prefilled array size [12k, 12k] multiplying into the first,
10 iterations
consistent result: 1.250 secs
it doesn't seem like its cheating, as number of iterations does not change prefill time(expected), and does change score time(also expected)
*/

import java.util.Arrays;
import java.util.Random;
public class Speedtest2 {
    public static final int xSize = 12000;
    public static final int ySize = 12000;
    public static final int iterations = 10;
    double[][] aArray = new double[ySize][xSize];
    double[][] bArray = new double[ySize][xSize];
    
    // constructor
    public Speedtest2(){
        for(int y = 0; y < ySize; y++) {
            for(int x = 0; x < xSize; x++){
                // prefill with random doubles
                aArray[y][x] = (new Random().nextDouble() * 2);
                bArray[y][x] = (new Random().nextDouble() * 2);
            }
        }
    }
    
    public void process() {
        for(int y = 0; y < ySize; y++) {
            for(int x = 0; x < xSize; x++){
                // multiply both into aArray
                aArray[y][x] *= bArray[y][x];
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
        Speedtest2 thisTest = new Speedtest2();
        
        // start timer
        long startTime = System.currentTimeMillis();
        System.out.println("started");
        
        // mult
        for(int i = 0; i < iterations; i++){
            thisTest.process();
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