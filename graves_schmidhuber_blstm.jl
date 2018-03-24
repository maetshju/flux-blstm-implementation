using Flux
using Flux: crossentropy, softmax, throttle, flip
using JLD

traindir = "train"

forward = LSTM(26, 93)
backward = LSTM(26, 93)
# output = Dense(186, 61)

blstm(x1, xn) = vcat(forward(x1), backward(xn))

# function model(xs)
#     x1, xn = xs
# #     Flux.reset!((forward, backward))
#     softmax(output(blstm(x1, xn)))
# #     yhat = softmax.(output.(blstm(x)))
# #     return yhat
# end

function BLSTM(n_in, n_out)
#     forward = LSTM(26, 93)
#     backward = LSTM(26, 93)
    x -> vcat(forward(x[1]), backward(x[2]))
end

model = Chain(
    BLSTM(26, 93),
    Dense(186, 61),
    softmax
)

# model = Chain(
#     Dense(26, 93),
#     Dense(93, 61),
#     softmax
# )

function loss(x, y)
    l = sum(crossentropy.(model.(collect(zip(x, reverse(x)))), y))
    Flux.reset!((forward, backward))
#     println(l / size(x,1))
#     Flux.reset!((forward, backward))
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

function get_prediction(y)
        mx = 0
        loc = 0
        for (i, value) in enumerate(y)
            if value > mx
                mx = value
                loc = i
            end
        end
        return loc
    end

println("Beginning training")

epochs = 1
for i in 1:epochs

    println("Epoch " * string(i) * "\t")
    data = data[shuffle(1:length(data))]
    
    Flux.train!(loss, data, opt)
    
    val_predictions = Vector()
    val_correct = Vector()

    println("Validating")
    for (x, y) in val_data
        utterancepredictions = model.(collect(zip(x, reverse(x))))
        utterance_correct = Vector()
        for (yhat, y_single) in zip(utterancepredictions, y)
            yhat = indmax(yhat) # COULD USE INDMAX HERE?
            y_single = get_prediction(y_single)
            push!(utterance_correct, yhat == y_single)
        end
        val_predictions = vcat(val_predictions, utterancepredictions)
        val_correct = vcat(val_correct, utterance_correct)
    end

    println("Val acc. " * string(sum(val_correct) / length(val_correct)))

    println()
end
