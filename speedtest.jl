# speed test
# qty 2 prefilled array size [12k, 12k] multiplying into the first,
# 20 iterations: 6360 6380 6390 6400 7110

const global SIZE_X = 12000
const global SIZE_Y = 12000
const global ITERATIONS = 20

function process(a, b)
    return a .* b
end

function processHandler(aArr, bArr)
    for iter = 1:ITERATIONS
        # process ITERATIONS num of times
        aArr = process(aArr, bArr)
    end
    return aArr
end

function main()
    # prefill with random float64 values from 0 - 2
    aArr = rand(Float64, (SIZE_X, SIZE_Y))
    bArr = rand(Float64, (SIZE_X, SIZE_Y))
    aArr *= 2.0
    bArr *= 2.0
    #println(aArr)
    #println(bArr)
    
    # start timer
    @time aArr = processHandler(aArr, bArr)
    
    #println(aArr)
end

main()