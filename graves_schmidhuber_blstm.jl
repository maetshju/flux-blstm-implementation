# blstm.jl
# 
# An implementation of the bidirectional LSTM neural net for
# speech recognition, as defind by Graves & Schmidhuber (2005).
# [[NEED FULL CITATION]].
# 
# Julia implementation copyright Matthew C. Kelley, 2018
# 
# This software is licensed under the MIT license.
# [[THIS MIGHT NOT BE DOABLE; CHECK JULIA AND FLUX LICENSES]]

using Flux
using Flux: crossentropy, softmax, throttle, flip
using JLD

traindir = "train"

forward = LSTM(26, 93)
backward = LSTM(26, 93)

BLSTM(x) = vcat(forward(x[1]), backward(x[2]))

# function BLSTM(n_in, n_out)
#     x -> vcat(forward(x[1]), backward(x[2]))
# end

model = Chain(
    BLSTM,
    Dense(186, 61),
    softmax
)

function loss(x, y)
    l = sum(crossentropy.(model.(collect(zip(x, reverse(x)))), y))
    Flux.reset!((forward, backward))
    return l
end

opt = SGD(params(model), 10.0^-5)

fnames = readdir("train")

Xs = Vector()
Ys = Vector()

println("Loading files")
for (i, fname) in enumerate(fnames)

    print(string(i) * "/" * string(length(fnames)) * "\r")
    x, y = load(joinpath(traindir, fname), "x", "y")
    x = [x[i,:] for i in 1:size(x,1)]
    y = [y[i,:] for i in 1:size(y,1)]
    push!(Xs, x)
    push!(Ys, y)
end

data = collect(zip(Xs, Ys))

val_data = data[1:184]
data = data[184:length(data)]

predict(x) = model.(collect(zip(x, reverse(x))))

function evaluate_accuracy(data)
    correct = Vector()
    for (x, y) in data
        y = indmax.(y)
        yhat = indmax.(predict(x))
        correct = vcat(correct, [yhat1 == y1 for (yhat1, y1) in zip(yhat, y)])
    end
    sum(correct) / length(correct)
end

println("Beginning training")

epochs = 1

for i in 1:epochs

    println("Epoch " * string(i) * "\t")
    data = data[shuffle(1:length(data))]
    
    Flux.train!(loss, data, opt)
    val_acc = evaluate_accuracy(val_data)

    print("Validating\r")
    println("Val acc. " * string(val_acc))
    println()
end
