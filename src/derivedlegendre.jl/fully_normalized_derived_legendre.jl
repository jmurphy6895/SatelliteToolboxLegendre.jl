export fully_normalized_derived_legendre
#TODO: FIX DOCS
"""
    fully_normalized_legendre!(P::AbstractMatrix{T}, ϕ::Number, n_max::Integer = -1, m_max::Integer = -1; kwargs...) where T<:Number -> Nothing

Compute the fully normalized associated Legendre function `P_n,m[cos(ϕ)]`. The maximum
degree and order that will be computed are given by the arguments `n_max` and `m_max`. If
they are negative (default), the dimensions of matrix `P` will be used:

    maximum degree -> number of rows - 1
    maximum order  -> number of columns - 1

The result will be stored in the matrix `P`.

# Keywords

- `ph_term::Bool`: If `true`, the Condon-Shortley phase term `(-1)^m` will be included.
    (**Default** = `false`)

# Remarks

This algorithm was based on **[1]**. Our definition of fully normalized associated Legendre
function can be seen in **[2, p. 546]**. The conversion is obtained by:

             ┌                       ┐
             │  (n-m)! . k . (2n+1)  │
    K_n,m = √│ ───────────────────── │,  k = (m = 0) ? 1 : 2.
             │         (n+m)!        │
             └                       ┘

    P̄_n,m = P_n,m * K_n,m,

where `P̄_n,m` is the fully normalized Legendre associated function.

# References

- **[1]** Holmes, S. A. and W. E. Featherstone, 2002. A unified approach to the Clenshaw
    summation and the recursive computation of very high degree and order normalised
    associated Legendre functions. Journal of Geodesy, 76(5), pp. 279-299. For more info.:
    http://mitgcm.org/~mlosch/geoidcookbook/node11.html

- **[2]** Vallado, D. A (2013). Fundamentals of Astrodynamics and Applications. Microcosm
    Press, Hawthorn, CA, USA.
"""
function fully_normalized_derived_legendre!(
    A::AbstractMatrix{T},
    u::Number,
    n_max::Integer = -1,
    m_max::Integer = -1;
) where T<:Number

    # Obtain the maximum degree and order that must be computed.
    n_max, m_max = _get_degree_and_order(A, n_max, m_max)

    # Get the first indices in `P` to take into account offset arrays.
    i₀, j₀ = first.(axes(A))

    sq3 = √T(3)

    A[i₀, j₀] = 1.0
    A[i₀+1, j₀] = u*sq3
    A[i₀+1, j₀+1] = sq3
    
    @inbounds for n in 2:min(n_max, m_max)
        A[i₀+n, j₀+n] = √((2*n + 1)/(2*n)) * A[i₀+n-1, j₀+n-1]
    end

    @inbounds for n in 2:n_max
        for m in 0:min(n-1, m_max)
            b_nm = √(((2.0*n+1.0)*(2.0*n-1.0)) / ((n+m)*(n-m)))
            b_n1m = √(((2.0*n-1)*(2.0*n-3.0)) / ((n+m-1.0)*(n-m-1.0)))

            A[i₀+n, j₀+m] = u*b_nm*A[i₀+n-1, j₀+m] - (b_nm/b_n1m)*A[i₀+n-2, j₀+m]
        end
    end

    return nothing

end

function fully_normalized_derived_legendre(
    u::T,
    n_max::Integer,
    m_max::Integer = -1;
) where T<:Number
    n_max < 0 && throw(ArgumentError("n_max must be positive."))

    if (m_max < 0) || (m_max > n_max)
        m_max = n_max
    end

    A = zeros(float(T), n_max + 1, m_max + 1)
    fully_normalized_derived_legendre!(A, u)

    return A
end
