/*
num of mult accs:
/
per member per evo step
12080 mults
81 adds(x % 150 === 0, add an add)
81 mults(squash fn, added to regular mults)
+conditional per x
-
10 evo steps, 6k pop:
gpu: 636ms
cpu: 12.7s
gpu wins at 20x faster or 5% time taken of cpu
better yet, the gpu going thru more evo steps doesn't really increase its time as it isn't saturated
so the gpu has an unknowable speed increase, but it's a fuckton

did speed test with c
CPU c took 5.736 sec
CPU node took 14.39 sec(2.5x slower than c)
GPU node took 1.22 sec(4.7x faster than cpu c, meaning context matters, obv. gpu in c would be immensely more complex and faster even than gpu node, but considering the conveniences of node, its a LOT faster for no other reason than I'm using parallel shit)
*/

// setup
import eco from './eco.3/eco.lib.js';
import chalk from 'chalk';
import { GPU } from 'gpu.js';
const gpu = new GPU();

const gpuSettings = {
    pipeline: true,
    dynamicOutput: true,
    output: {
        x: 16000,
    },
};


const gpumult = gpu.createKernel(function(){
    return((1.78939423494 * 0.785186454) + (4.113810156 * .500005123));
}, gpuSettings);

const cpumult = () => {
    let retArray = [];
    for(let x = 0; x < gpuSettings.output.x; x++){
        retArray.push((1.78939423494 * 0.785186454) + (4.113810156 * .500005123));
    };
    return(retArray);
};


// MAIN
const mainLoop = async _ => {

    const evoSteps = 100000;

   
   console.time("time");
   for(let i = 0; i < evoSteps; i++){ // must comment out resCpu or resGpu
        //let resCpu = cpumult();
        let resGpu = gpumult();
    };
    console.timeEnd("time");

    eco.loop.unlock();
};
eco.loop.nforever(mainLoop, 100);