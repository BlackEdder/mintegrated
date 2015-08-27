module mintegrated;

import std.array : array;
import std.conv : to;
import std.range : iota, zip;
import std.algorithm : all, each, map, max, reduce, filter;
import std.random : uniform;
import std.math : pow, round, sqrt;

import scid.types : Result;
import dstats.summary : meanStdev, MeanSD;

private struct Area(Real)
{
    Real[] lower;
    Real[] upper;
}

private Real volume(Real)( in Area!Real area )
{
    /+
        D way: fails to compile, so use for loop
        return zip(area.lower, area.upper)
        .map((t) => t[1] - t[0])
        .reduce!((a,b) => a*b );
    +/
    Real s = 1;
    foreach( t; zip(area.lower, area.upper) )
        s *= t[1] - t[0];
    return s;
}

unittest
{
    assert( volume( Area!double([-1.0,-2.0],[2.0,0.0]) ) == 3.0*2.0 );
}

private Area!Real[] splitArea(Real)( in Area!Real area, size_t dimension )
{
    assert( dimension < area.lower.length );
    auto div = (area.upper[dimension]-area.lower[dimension])*0.5 +
        area.lower[dimension];
    auto newLower = area.lower.dup;
    newLower[dimension] = div;
    auto newUpper = area.upper.dup;
    newUpper[dimension] = div;
    return [ Area!Real( area.lower.dup, newUpper ),
           Area!Real( newLower, area.upper.dup ) ];
}

unittest
{
    auto a = Area!double([-1.0,-2.0],[2.0,0.0]);
    assert( splitArea(a,0)[0].volume == 0.5*a.volume );
    assert( splitArea(a,1)[0].volume == 0.5*a.volume );
    assert( splitArea(a,0)[1].volume == 0.5*a.volume );
    assert( splitArea(a,1)[1].volume == 0.5*a.volume );
    assert( splitArea(a,0)[1].lower[0] == 0.5 );
    assert( splitArea(a,1)[1].lower[0] == -1.0 );
}

private bool withinArea(Real)( in Real[] point, in Area!Real area )
{
    return zip(point, area.lower, area.upper).all!(
            (t) => t[0] >= t[1] && t[0] <= t[2] );
}

unittest
{
    auto a = Area!double([-1.0,-2.0],[2.0,0.0]);
    assert( [0.0,0.0].withinArea( a ) );
    assert( ![-2.0,0.0].withinArea( a ) );
    assert( ![3.0,0.0].withinArea( a ) );
    assert( ![0.0,-2.1].withinArea( a ) );
    assert( ![0.0,0.1].withinArea( a ) );
}

private Result!Real meanAndVariance(Real, Range : MeanSD)( in Range msd, in Area!Real area )
{
    auto v = area.volume;
    return Result!Real( v*msd.mean().to!Real,
            pow(v,2)*msd.mse().to!Real );
}

private Result!Real meanAndVariance(Real, Range)( in Range values, in Area!Real area )
{
    auto msd = meanStdev( values );
    return msd.meanAndVariance!Real( area );
}


unittest
{
    auto a = Area!double([-1.0,-2.0],[0.0,-1.0]);
    auto vs = [1.0,2.0,1.5];
    auto res = vs.meanAndVariance( a );
    assert( res.value == 1.5 );
    assert( res.error == 0.5/3 );

    a = Area!double([-1.0,-2.0],[1.0,-1.0]);
    res = vs.meanAndVariance( a );
    assert( res.value == 2*1.5 );
    assert( res.error == 4*0.5/3 );
}

Result!Real integrate(Func, Real)(scope Func f, Real[] a, Real[] b,
    Real epsRel = cast(Real) 1e-6, Real epsAbs = cast(Real) 0)
{
    return miser(f, a, b, epsRel, epsAbs, 150*a.length); 
}

Result!Real miser(Func, Real)(scope Func f, Real[] a, Real[] b,
    Real epsRel = cast(Real) 1e-6, Real epsAbs = cast(Real) 0, 
    size_t npoints = 1000 )
{
    import std.stdio : writeln;

    auto area = Area!Real( a, b );
    auto bounds = a.zip(b);
    auto points =
        iota( 0, npoints, 1 )
        .map!( (i) => bounds
                .map!( (t) {
                    return uniform!"()"( t[0], t[1] ).to!Real; 
                    } 
                ).array 
             );
    auto values = points.map!((pnt) => f( pnt ) ).array;

    auto result = meanAndVariance( values, area );

    auto error = sqrt( result.error );
    if (error < epsAbs || error/result.value < epsRel)
        return Result!Real(result.value, error);


    // Try different subareas

    Area!Real[] bestAreas;
    auto bestEst = Real.max;
    Result!Real[] bestResults;
    foreach( j; 0..a.length ) 
    {
        auto subAreas = area.splitArea( j );

        auto pntvs = zip( points, values );
        auto results = subAreas.map!( (a) 
                {
                    MeanSD msd;
                    foreach( pntv; zip( points, values ) )
                        {
                        if (pntv[0].withinArea(a))
                        msd.put( pntv[1] );
                        }
                    assert( msd.N > 0 );
                    return msd.meanAndVariance( a );
                } );
        Result!Real[] cacheResults;
        // Optimize this by first only looking at first. Only if that
        // is smaller than bestEstimate would we need to calculate second
        Real runningError = 0;
        while( !results.empty && runningError < bestEst )
        {
            runningError += results.front.error;
            cacheResults ~= results.front;
            results.popFront;
        }

        if( results.empty && runningError < bestEst )
        {
            bestEst = runningError;
            bestResults = cacheResults;
            bestAreas = subAreas;
        }
    }
    assert( bestAreas.length == 2 );
    assert( bestResults.length == 2 );

    auto sdA = sqrt(bestResults[0].error);
    auto sdB = sqrt(bestResults[1].error);
    auto sumSd = sdA + sdB;
    assert( volume(bestAreas[0]) > 0 );
    if (sumSd == 0)
    {
        result = Result!Real( (bestResults[0].value+bestResults[1].value), 
            sqrt(0.25*pow(bestResults[0].error,2)+0.25*pow(bestResults[1].error,2) ) );

        return result;
    }

    auto npntsl = round(150*a.length*sdA/sumSd).to!int; 
    auto npntsu = 150*a.length-npntsl;

    auto rl = miser( f, bestAreas[0].lower, bestAreas[0].upper, 
            epsRel, epsAbs,
            max( 15*a.length, npntsl ) );
    auto ru = miser( f, bestAreas[1].lower, bestAreas[1].upper, 
            epsRel, epsAbs,
            max( 15*a.length, npntsu ) );

    result = Result!Real( (rl.value+ru.value), 
            sqrt(0.25*pow(rl.error,2)+0.25*pow(ru.error,2) ) );

    return result; 
}

///
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

    auto result = integrate( func, [-1.0,-1], [1.0,1.0], 1e-5, 1e-3 );
    result.writeln;
    assert( result.value <= PI + 1e-3 );
    assert( result.value >= PI - 1e-3 );
}
