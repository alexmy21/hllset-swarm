"""
MIT License

Copyright (c) 2023: Jakob Nybo Nissen.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

https://github.com/jakobnissen/Probably.jl/blob/master/src/hyperloglog/hyperloglog.jl

"""

"""
HyperLogLog Set Implementation with Extended Set Operations

This module provides a HyperLogLog (HLL) implementation with additional set operations
including union, intersection, difference, and complement operations.

Key Features:
- Efficient cardinality estimation using HLL algorithm
- Set operations between HLL sets
- Serialization/deserialization support
- Similarity measures (Jaccard, Cosine)
"""

module HllSets

    include("constants.jl")
    using SHA 

    export HllSet, add!, count, union, intersect, diff, isequal, isempty, id, delta, getbin, getzeros, maxidx, match, cosine, dump, restore, to_binary_tensor, flatten_tensor, tensor_to_string, string_to_tensor, binary_tensor_to_hllset, set_xor, set_comp, set_added, set_deleted

    struct HllSet{P}
        counts::Vector{UInt32}

        function HllSet{P}() where {P}
            isa(P, Integer) || throw(ArgumentError("P must be integer"))
            (P < 4 || P > 18) && throw(ArgumentError("P must be between 4 and 18"))
            new(fill(UInt32(0), 2^P))
        end
    end

    function HllSet(p::Int=10)
        return HllSet{p}()
    end

    # Core HLL Operations --------------------------------------------------------

    """
        add!(hll::HllSet{P}, x::Any; seed::Int=0)

    Add an element to the HLL set.
    """
    function add!(hll::HllSet{P}, x::Any; seed::Int = 0) where {P}
        # println("seed = ", seed, "; P = ", P, "; x = ", x)
        h = u_hash(x; seed=seed)
        # println("hash = ", h)
        bin = getbin(hll, h)
        idx = getzeros(hll, h)
        if idx <= 32
            hll.counts[bin] |= (1 << (idx - 1))
        end
    end

    function add!(hll::HllSet{P}, values::Union{Set, Vector}; seed::Int = 0) where {P}
        for value in values
            add!(hll, value, seed=seed)
        end
    end    

    # Helper Functions ----------------------------------------------------------

    function _validate_compatible(x::HllSet{P}, y::HllSet{P}) where {P}
        length(x.counts) == length(y.counts) || 
            throw(ArgumentError("HLL sets must have same precision"))
    end

    # Set Operations ------------------------------------------------------------

    """
        union(x::HllSet{P}, y::HllSet{P}) where {P}

    Compute union of two HLL sets.
    """
    function Base.union!(dest::HllSet{P}, src::HllSet{P}) where {P}
        _validate_compatible(dest, src)

        @inbounds for i in 1:length(dest.counts)
            dest.counts[i] = dest.counts[i] .| src.counts[i]
        end
        return dest
    end

    function Base.union(x::HllSet{P}, y::HllSet{P}) where {P} 
        _validate_compatible(x, y)

        z = HllSet{P}()
        @inbounds for i in 1:length(x.counts)
            z.counts[i] = x.counts[i] .| y.counts[i]
        end
        return z
    end

    """
        intersect(x::HllSet{P}, y::HllSet{P}) where {P}

    Compute intersection of two HLL sets.
    """
    function Base.intersect(x::HllSet{P}, y::HllSet{P}) where {P} 
        _validate_compatible(x, y)

        z = HllSet{P}()
        @inbounds for i in 1:length(x.counts)
            z.counts[i] = x.counts[i] .& y.counts[i]
        end
        return z
    end

    """
        diff(hll_1::HllSet{P}, hll_2::HllSet{P}) where {P}

    Compute difference between two HLL sets.
    """
    function Base.diff(hll_1::HllSet{P}, hll_2::HllSet{P}) where {P}
        _validate_compatible(hll_1, hll_2)
        
        n = HllSet{P}()
        d = HllSet{P}()
        r = HllSet{P}()

        d = set_comp(hll_1, hll_2)
        n = set_comp(hll_2, hll_1)
        r = intersect(hll_1, hll_2)

        return (DEL = d, RET = r, NEW = n)
    end

    """
        set_comp(x::HllSet{P}, y::HllSet{P}) where {P}

    Compute difference between two HLL sets as a y compliment to x. Return y elements that are not in x.
    """

    function set_comp(x::HllSet{P}, y::HllSet{P}) where {P} 
        _validate_compatible(x, y)

        z = HllSet{P}()
        @inbounds for i in 1:length(x.counts)
            z.counts[i] = .~y.counts[i] .& x.counts[i]
        end
        return z
    end

    """
        set_added(x::HllSet{P}, y::HllSet{P}) where {P}

    Compute difference between two HLL sets as a x compliment to y. Return x elements that are not in y.
    """
    function set_xor(x::HllSet{P}, y::HllSet{P}) where {P} 
        length(x.counts) == length(y.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        z = HllSet{P}()
        @inbounds for i in 1:length(x.counts)
            z.counts[i] = xor.(x.counts[i], (y.counts[i]))
        end
        return z
    end

    """
        copy(dest::HllSet{P}, src::HllSet{P}) where {P}

    creates copy of HllSet{P} object from src to dest
    """
    function Base.copy!(dest::HllSet{P}, src::HllSet{P}) where {P}
        length(dest.counts) == length(src.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        @inbounds for i in 1:length(dest.counts)
            dest.counts[i] = src.counts[i]
        end
        return dest
    end

    function Base.copy!(src::HllSet{P}) where {P}
        # length(dest.counts) == length(src.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        dest = HllSet{P}()
        @inbounds for i in 1:length(src.counts)
            dest.counts[i] = src.counts[i]
        end
        return dest
    end  

    """
        isequal(x::HllSet{P}, y::HllSet{P}) where {P}

    Check if two HLL sets are equal.
    """    
    function Base.isequal(x::HllSet{P}, y::HllSet{P}) where {P} 
        length(x.counts) == length(y.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        @inbounds for i in 1:length(x.counts)
            x.counts[i] == y.counts[i] || return false
        end
        return true
    end    

    Base.isempty(x::HllSet{P}) where {P} = all(all, x.counts)   

    """
        count(x::HllSet{P}) where {P}

    Estimate the cardinality of the HLL set.
    """
    function Base.count(x::HllSet{P}) where {P}
        # Harmonic mean estimates cardinality per bin. There are 2^P bins
        harmonic_mean = sizeof(x) / sum(1 / 1 << maxidx(i) for i in x.counts)
        biased_estimate = α(x) * sizeof(x) * harmonic_mean
        return round(Int, biased_estimate - bias(x, biased_estimate))
    end

    """
        Set of helper functions for cardinality estimation.
    """
    function getbin(hll::HllSet{P}, x::Int) where {P}
        return getbin(x, P=P)        
    end

    function getbin(x::Int; P::Int=10) 
        # Increasing P by 1 to compensate BitVector size that is of size 64
        x = x >>> (8 * sizeof(UInt) - (P + 1)) + 1
        str = replace(string(x, base = 16), "0x" => "")
        return parse(Int, str, base = 16)
    end

    function getzeros(hll::HllSet{P}, x::Int) where {P}
        return getzeros(x, P=P)
    end

    function getzeros(x::Int; P::Int=10)
        or_mask = ((UInt(1) << P) - 1) << (8 * sizeof(UInt) - P)
        return trailing_zeros(x | or_mask) + 1
    end
    α(x::HllSet{P}) where {P} =
        if P == 4
            return 0.673
        elseif P == 5
            return 0.697
        elseif P == 6
            return 0.709
        else
            return 0.7213 / (1 + 1.079 / sizeof(x))
        end 
    
    function bias(::HllSet{P}, biased_estimate) where {P}
        # For safety - this is also enforced in the HLL constructor
        if P < 4 || P > 18
            error("We only have bias estimates for P ∈ 4:18")
        end
        rawarray = @inbounds RAW_ARRAYS[P - 3]
        biasarray = @inbounds BIAS_ARRAYS[P - 3]
        firstindex = searchsortedfirst(rawarray, biased_estimate)
        # Raw count large, no need for bias correction
        if firstindex == length(rawarray) + 1
            return 0.0
            # Raw count too small, cannot be corrected. Maybe raise error?
        elseif firstindex == 1
            return @inbounds biasarray[1]
            # Else linearly approximate the right value for bias
        else
            x1, x2 = @inbounds rawarray[firstindex - 1], @inbounds rawarray[firstindex]
            y1, y2 = @inbounds biasarray[firstindex - 1], @inbounds biasarray[firstindex]
            delta = @fastmath (biased_estimate - x1) / (x2 - x1) # relative distance of raw from x1
            return y1 + delta * (y2 - y1)
        end
    end

    function maxidx(x::UInt32)        
        total_bits = sizeof(x) * 8
        leading_zeros_count = leading_zeros(x)
        return total_bits - leading_zeros_count
    end

    # Match Operations ------------------------------------------------------------

    """
        match(x::HllSet{P}, y::HllSet{P}) where {P}
    Compute the Jaccard similarity between two HLL sets.
    """
    function Base.match(x::HllSet{P}, y::HllSet{P}) where {P}
        length(x.counts) == length(y.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        
        count_u = count(union(x, y))
        count_i = count(intersect(x, y))
        return round(Int64, ((count_i / count_u) * 100))
    end

    """
        cosine(hll_1::HllSet{P}, hll_2::HllSet{P}) where {P}
    Compute the cosine similarity between two HLL sets.
    """
    function cosine(hll_1::HllSet{P}, hll_2::HllSet{P}) where {P}
        length(hll_1.counts) == length(hll_2.counts) || throw(ArgumentError("HllSet{P} must have same size"))

        v1 = hll_1.counts
        v2 = hll_2.counts
        if norm(v1) == 0 || norm(v2) == 0
            return 0.0
        end
        return dot(v1, v2) / (norm(v1) * norm(v2))
    end

    # Serialization and Deserialization ----------------------------------------

    """
        to_binary_tensor(x::HllSet{P}) where {P}
    Convert the HLL set to a binary tensor.
    """
    function to_binary_tensor(hll::HllSet{P}) where {P}
        tensor = zeros(Bool, 2^P, 32)
        for i in 1:2^P
            binary_str = bitstring(hll.counts[i])
            for j in 1:32
                tensor[i, j] = binary_str[j] == '1'
            end
        end
        return tensor
    end

    """
        flatten_tensor(tensor::Array{Bool, 2})
    Flatten the binary tensor to a 1D array.
    """
    function flatten_tensor(tensor::Array{Bool, 2})
        return vec(tensor)
    end

    """
        tensor_to_string(x::HllSet{P}) where {P}
    Convert the binary tensor to a string representation.
    """
    function tensor_to_string(flattened_tensor::Vector{Bool})
        # return join(flattened_tensor .== true ? "1" : "0", "")
        return join([b ? "1" : "0" for b in flattened_tensor], "")
    end

    """
        string_to_tensor(str::String, P::Int=10)
    Convert a string representation of a binary tensor to a 2D array.
    """
    function string_to_tensor(str::String, P::Int=10)
        str = rpad(str, 2^P * 32, "0")
        @assert length(str) == 2^P * 32 "String length must be 2^P * 32"
        return reshape([parse(Bool, c) for c in str], (2^P, 32))
    end

    """
        binary_tensor_to_hllset(tensor::Array{Bool, 2}, P::Int=10)
    Convert a binary tensor to an HLL set.
    """
    function binary_tensor_to_hllset(tensor::Array{Bool, 2}, P::Int=10)
        hll = HllSet{P}()
        for i in 1:2^P
            hll.counts[i] = parse(UInt32, tensor_to_string(tensor[i, :]), base=2)
        end
        return hll
    end   
   
    # HllSet ID ---------------------------------------------------
    
    """
        id(x::HllSet{P}) where {P}
    Compute the ID of the HLL set.
    """
    function id(x::HllSet{P}) where {P}
        if x == nothing
            return nothing
        end
        # Convert the Vector{UInt32} to a byte array
        bytearray = reinterpret(UInt8, x.counts)

        # Calculate the SHA1 hash
        hash_value = SHA.sha1(bytearray)
        return SHA.bytes2hex(hash_value)
    end

    function sha1(x::HllSet{P}) where {P}
        # Convert the Vector{BitVector} to a byte array
        bytearray = UInt8[]
        for bv in x.counts
            append!(bytearray, reinterpret(UInt8, bv))
        end
        # Calculate the SHA1 hash
        hash_value = SHA.sha1(bytearray)
        return SHA.bytes2hex(hash_value)
    end

    function u_hash(x; seed::Int=0) 
        if seed == 0
            abs_hash = abs(hash(x))
        else
            abs_hash = abs(hash(hash(x) + seed))
        end         
        return Int(abs_hash % typemax(Int64))
    end

    # Overload the show function to print the HllSet --------------------------------------------------
    Base.show(io::IO, x::HllSet{P}) where {P} = println(io, "HllSet{$(P)}()")

    Base.sizeof(::Type{HllSet{P}}) where {P} = 1 << P
    Base.sizeof(x::HllSet{P}) where {P} = sizeof(typeof(x))

    # Depricated Functions --------------------------------------------------
    
    function Base.dump(x::HllSet{P}) where {P}
        # Base.depwarn("dump(hll::HllSet{P}) is deprecated, use getcounts(x::Int; P::Int=10) instead.", :getbin)
        # For safety - this is also enforced in the HLL constructor
        if P < 4 || P > 18
            error("We only have dump for P ∈ 4:18")
        end
        
        return x.counts
    end

    function restore!(z::HllSet{P}, x::Vector{UInt32}) where {P} 
        # For safety - this is also enforced in the HLL constructor
        if P < 4 || P > 18
            error("We only have restore for P ∈ 4:18")
        end
        if length(x) != length(z.counts)
            error("The length of the vector must be equal to the length of the HllSet")
        end        
        # z.counts = x
        @inbounds for i in 1:length(x)
            z.counts[i] = x[i]
        end
        return z
    end

    function restore!(z::HllSet{P}, x::String) where {P}
        # For safety - this is also enforced in the HLL constructor
        if P < 4 || P > 18
            error("We only have restore for P ∈ 4:18")
        end
        dataset = JSON3.read(x, Vector{UInt32})
        
        @inbounds for i in 1:length(x)
            z.counts[i] = x[i]
        end
        return z
    end 

end
