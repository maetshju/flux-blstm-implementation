using Flux: onehotbatch
using WAV
using MFCC
using PyCall
using JLD

training_data_dir = "TIMIT/TRAIN"
test_data_dir = "TIMIT/TEST"

training_out_dir = "train"
test_out_dir = "test"

phones = split("h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng")
phone_translations = Dict(phone=>i for (i, phone) in enumerate(phones))

function make_data(phn_fname, wav_fname)

    samps, sr = wavread(wav_fname)
    samps = reshape(samps, size(samps)[1])
    frames, _, _ = mfcc(samps, sr, :rasta; wintime=0.025, steptime=0.01)
    frame_deltas = deltas(frames)
    features = hcat(frames, frame_deltas)
    
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
    
    idx = 1
    label = labels[idx]
    boundary = boundaries[idx]
    
    winlen_samps = 0.025 * sr
    steplen_samps = 0.01 * sr
    
    seq = Vector()
    
    for i=1:size(features)[1]
        win_end = winlen_samps + (i-1)*steplen_samps
        # check if more than half of the samples are in the next label
        if idx < length(boundaries) && win_end - boundary > winlen_samps / 2
            idx += 1
            label = labels[idx]
            boundary = boundaries[idx]
        end
        
        push!(seq, label)
    end
    
    (features, seq)
end

# create training data files
for (root, dirs, files) in walkdir(training_data_dir)
    phn_fnames = [fname for fname in files if contains(fname, "PHN")]
    wav_fnames = [fname for fname in files if contains(fname, "WAV")]
    
    one_dir_up = basename(root)
    println(root)
    
    for (phn_fname, wav_fname) in zip(phn_fnames, wav_fnames)
        phn_path = joinpath(root, phn_fname)
        wav_path = joinpath(root, wav_fname)
        x, y = make_data(phn_path, wav_path)
        
        y = [phone_translations[Y] for Y in y]
        class_nums = [x for x in 1:61]
        y = onehotbatch(y, class_nums)'
        
        base = splitext(phn_fname)[1]
        dat_name = one_dir_up * base * ".jld"
        dat_path = joinpath(training_out_dir, dat_name)
        save(dat_path, "x", x, "y", y)
    end
end


# create testing data files
for (root, dirs, files) in walkdir(test_data_dir)
    phn_fnames = [fname for fname in files if contains(fname, "PHN")]
    wav_fnames = [fname for fname in files if contains(fname, "WAV")]
    
    one_dir_up = basename(root)
    println(root)
    
    for (phn_fname, wav_fname) in zip(phn_fnames, wav_fnames)
        phn_path = joinpath(root, phn_fname)
        wav_path = joinpath(root, wav_fname)
        x, y = make_data(phn_path, wav_path)
        
        y = [phone_translations[Y] for Y in y]
        class_nums = [x for x in 1:61]
        y = onehotbatch(y, class_nums)
        
        x = [x[i,:] for i in 1:size(x,1)]
        y = [y[i,:] for i in 1:size(y,1)]
        
        base = splitext(phn_fname)[1]
        dat_name = one_dir_up * base * ".jld"
        dat_path = joinpath(test_out_dir, dat_name)
        save(dat_path, "x", x, "y", y)
    end
end
