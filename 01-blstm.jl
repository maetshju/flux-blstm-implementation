# blstm.jl
# 
# An implementation of the bidirectional LSTM neural net for
# speech recognition defind by Graves & Schmidhuber (2005).
# [[NEED FULL CITATION]].
# 
# Julia implementation copyright Matthew C. Kelley, 2018
# 
# This software is licensed under the MIT license.
# [[THIS MIGHT NOT BE DOABLE; CHECK JULIA AND FLUX LICENSES]]

using Flux
using Flux: crossentropy, softmax, throttle, flip, sigmoid
using JLD

# Paths to the training and test data directories
traindir = "train"
testdir = "test"

# Component layers of the bidirectional LSTM layers
forward = LSTM(26, 93)
backward = LSTM(26, 93)

function BLSTM(x)
    # BLSTM layer using above LSTM layers
    #
    # Parameters
    #   x A 2-tuple containing the forward and backward time
    #       samples; the first is from processing the sequence
    #       forward, and the second is from processing it backward
    #
    # Returns The concatenation of the forward and backward
    #   LSTM predictions
    sigmoid.(vcat(forward(x[1]), backward(x[2])))
end

# The model that is used for predicitons; consists of the
# BLSTM layer and and output layer 
model = Chain(
    BLSTM,
    Dense(186, 61),
    softmax
)

function loss(x, y)
    # Calculates the categorical cross-entropy loss for an utterance
    #
    # Parameters
    #   x Iterable containing the frames to classify
    #   y Iterable containing the labels corresponding to the frames
    #       in x
    #
    # Returns the calculated loss value
    #
    # Side-effects
    #   Resets the state in the BLSTM layer
    l = sum(crossentropy.(model.(collect(zip(x, reverse(x)))), y))
    Flux.reset!((forward, backward))
    return l
end

function read_data(data_dir)
    # Reads in the data contained in a specified directory
    #
    # Parameters
    #   data_dir String of the path to the directory containing the
    #       data
    #
    # Return
    #   Xs A vector where each element is a vector of the frames for
    #       one utterance
    #   Ys A vector where each element is a vector of the labels for
    #       the frames for one utterance
    fnames = readdir(data_dir)

    Xs = Vector()
    Ys = Vector()
    
    for (i, fname) in enumerate(fnames)
        print(string(i) * "/" * string(length(fnames)) * "\r")
        x, y = load(joinpath(traindir, fname), "x", "y")
        x = [x[i,:] for i in 1:size(x,1)]
        y = [y[i,:] for i in 1:size(y,1)]
        push!(Xs, x)
        push!(Ys, y)
    end
    
    return (Xs, Ys)
end

function predict(x)
    # Make predictions on the data using the model defined above
    #
    # Parameters
    #   x An iterable containing the frames for a single utterance
    #
    # Returns The predicted scores for each phoneme class for each
    #   frame in x
    #
    # Side effects
    #   Resets the state in the BLSTM layer
    model.(collect(zip(x, reverse(x))))
    Flux.reset!((forward, backward))
end

function evaluate_accuracy(data)
    # Evaluates the accuracy of the model on a set of data; can be
    # used either for validation or test accuracy
    #
    # Parameters
    #   data An iterable of paired values where the first element is
    #       all the frames for a single utterance, and the second is
    #       the associated frame labels to compare the model's
    #       predictions against
    3
    # Returns the predicted accuracy value as a proportion of the
    #   number of correct predictions over the total number of
    #   predictions made
    correct = Vector()
    for (x, y) in data
        y = indmax.(y)
        ŷ = indmax.(predict(x))
        correct = vcat(correct,
                        [ŷ_n == y_n for (ŷ_n, y_n) in zip(ŷ, y)])
    end
    sum(correct) / length(correct)
end

println("Loading files")
Xs, Ys = read_data(traindir)
data = collect(zip(Xs, Ys))

# Move 5% (184 files) of the TIMIT data into a validation set
val_data = data[1:184]
data = data[184:length(data)]

# Begin training
println("Beginning training")

opt = SGD(params(model), 10.0^-5)
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

# Test data
test_data = collect(zip(read_data(testdir)))
print("Testing\r")
test_acc = evaluate_accuracy(test_data)
println("Test acc. " * string(test_acc))
println()
