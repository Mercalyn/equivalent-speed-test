/*
speed test
qty 2 prefilled array size [12k, 12k] multiplying into the first,
20 iterations
consistent result: 
cpu: 3280 3300 3310 3310 3320
gpu: 960 1040 1130 1140 1310
*/

// setup
import eco from './eco.3/eco.lib.js';
import chalk from 'chalk';
import { GPU } from 'gpu.js';
const gpu = new GPU();

const iterations = 20;
const gpuSettings = {
    pipeline: true,
    dynamicOutput: true, // do NOT set immutable to true, will crash gpu high MEM useage
    output: {
        x: 12000,
        y: 12000,
    },
};


const prefillKernelFn = function(){
    // prefilling with gpu then converting to 2d array for cpu cause gpu waaaaay easier
    return((Math.random() * 2) + .0001); // .0001 to 2.000
};
const [prefillA, prefillB] = [
    gpu.createKernel(prefillKernelFn, gpuSettings),
    gpu.createKernel(prefillKernelFn, gpuSettings)
];


const gpumult = gpu.createKernel(function(textureA, textureB){
    return(textureA[this.thread.y][this.thread.x] * textureB[this.thread.y][this.thread.x]);
}, gpuSettings);

const cpumult = (cpuA, cpuB) => {
    for(let y = 0; y < gpuSettings.output.y; y++){
        for(let x = 0; x < gpuSettings.output.x; x++){
            cpuA[y][x] *= cpuB[y][x];
        };
    };
    return(cpuA);
};


// MAIN
const mainLoop = async _ => {
    // prefill
    let textureA = prefillA();
    let textureB = prefillB();
    /*
    let cpuA = textureA.toArray(); // if testing GPU, you MUST comment toArray out
    let cpuB = textureB.toArray();
    console.log(cpuA[0][0]);
    console.log(cpuB[0][0]);
    */
    
    
    // timer
    console.time("time");
    for(let i = 0; i < iterations; i++){ // must comment out resCpu or resGpu
        //cpuA = cpumult(cpuA, cpuB);
        let resGpu = gpumult(textureA, textureB);
    };
    console.timeEnd("time");
    
    
    // spot check
    //console.log(cpuA[0][0]); // cpu
    //console.log(textureA.toArray()[0][0]); // gpu
    //cpuA = cpumult(cpuA, cpuB);
    
    // cleanup
    textureA.delete();
    textureB.delete();
    
    eco.loop.unlock();
};
eco.loop.nforever(mainLoop, 100);