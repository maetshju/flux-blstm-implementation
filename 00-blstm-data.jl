# 00-blstm-data.jl
#
# Julia implementation Copyright (c) 2018 Matthew C. Kelley
#
# Script to pre-process TIMIT data for use in the neural network.

using Flux: onehotbatch
using WAV
using MFCC
using JLD

# Set up path names for reading and writing data
training_data_dir = "TIMIT/TRAIN"
test_data_dir = "TIMIT/TEST"

training_out_dir = "train"
test_out_dir = "test"

frame_length = 0.025 # 25 ms
frame_interval = 0.01 # 10 ms

# Establish mapping from TIMIT phoneme to category number
phones = split("h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng")
phone_translations = Dict(phone=>i for (i, phone) in enumerate(phones))

"""
    make_data(phn_fname, wav_fname)
Process a TIMIT WAV file into MFCC windows, labeled based on the PHN file.

# Parameters:
* **phn_fname** String of the PHN file name
* **wav_fname** String of the WAV file name
    
# Returns
* A tuple containing a vector of frames and a vector of labels
"""
function make_data(phn_fname, wav_fname)

    # Read in WAV file and process into MFCCs and deltas
    samps, sr = wavread(wav_fname)
    samps = reshape(samps, size(samps)[1])
    frames, _, _ = mfcc(samps,
                        sr,
                        :rasta;
                        wintime=frame_length,
                        steptime=frame_interval)
    frame_deltas = deltas(frames)
    features = hcat(frames, frame_deltas)
    
    # Obtain boundary and label information from PHN file
    lines = Vector()
    open(phn_fname, "r") do f
        
        lines = readlines(f)
    end
    
    boundaries = Vector()
    labels = Vector()
    
    for line in lines
        _, boundary, label = split(line)
        boundary = parse(Int64, boundary)
        
        push!(boundaries, boundary)
        push!(labels, label)
    end
    
    # Step through each frame in the signal, and assign a label
    # based on the processed PHN file
    idx = 1
    label = labels[idx]
    boundary = boundaries[idx]
    
    framelen_samps = frame_length * sr
    frameinterval_samps = frame_interval * sr
    
    seq = Vector()
    
    for i=1:size(features)[1]
    
        win_end = framelen_samps + (i-1)*frameinterval_samps
        
        # check if more than half of the samples are in the next label;
        # if so, start assigning next label
        if idx < length(boundaries) && win_end - boundary > framelen_samps / 2
            idx += 1
            label = labels[idx]
            boundary = boundaries[idx]
        end
        
        push!(seq, label)
    end
    
    (features, seq)
end

"""
    create_data(data_dir, out_dir)

Pre-processes the TIMIT data and writes it to the appropriate directories.

# Parameters
* **data_dir** The directory containing the data to pre-process
* **out_dir** The directory to write the pre-processed data to
"""
function create_data(data_dir, out_dir)
    for (root, dirs, files) in walkdir(data_dir)
        phn_fnames = [fname for fname in files if contains(fname, "PHN")]
        wav_fnames = [fname for fname in files if contains(fname, "WAV")]
        
        one_dir_up = basename(root)
        println(root)
        
        for (phn_fname, wav_fname) in zip(phn_fnames, wav_fnames)
            phn_path = joinpath(root, phn_fname)
            wav_path = joinpath(root, wav_fname)
            x, y = make_data(phn_path, wav_path)
            
            y = [phone_translations[Y] for Y in y]
            class_nums = [n for n in 1:61]
            y = onehotbatch(y, class_nums)'
            
            base = splitext(phn_fname)[1]
            dat_name = one_dir_up * base * ".jld"
            dat_path = joinpath(out_dir, dat_name)
            save(dat_path, "x", x, "y", y)
        end
    end
end

create_data(training_data_dir, training_out_dir)
create_data(test_data_dir, test_out_dir)
