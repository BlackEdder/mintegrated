# Mintegrated [![Build Status](https://travis-ci.org/BlackEdder/mintegrated.svg?branch=master)](https://travis-ci.org/BlackEdder/mintegrated)

Library for (multi)dimensional integration written in D. Currently limited to an implementation of the MISER algorithm

```D
unittest
{
    import std.math : PI, pow;
    import std.stdio : writeln;
    auto func = function( double[] xs )
    {
        if (pow(xs[0],2)+pow(xs[1],2)<= 1.0)
            return 1.0;
        return 0.0;
    };

    auto result = integrate( func, [-1.0,-1], [1.0,1.0], 1e-5, 1e-7 );
    result.writeln;
    assert( result.value <= PI + 1e-3 );
    assert( result.value >= PI - 1e-3 );
}
```
