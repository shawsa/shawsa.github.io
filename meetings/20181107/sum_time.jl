using BenchmarkTools
x = rand(10^7);
result = @btime sum(x)
#result = @elapsed sum(x);
println(result)
