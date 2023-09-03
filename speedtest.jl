# speed test
# qty 2 prefilled array size [12k, 12k] multiplying into the first,
# 20 iterations: 2630 2660 2670 2700 2720

const global SIZE_X = 12000
const global SIZE_Y = 12000
const global ITERATIONS = 20

#note that the func! means it modifies one or more of its arguments, instead of passing in a pointer prob
function process!(aArr, bArr)
    # process ITERATIONS num of times
    for iter = 1:ITERATIONS
        # element-wise mult
        aArr .*= bArr
    end
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
    @time process!(aArr, bArr)
    
    #println(aArr)
end

main()