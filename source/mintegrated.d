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

private size_t dimension(Real)( in Area!Real area )
{
    return area.lower.length;
}

unittest
{
    assert( volume( Area!double([-1.0,-2.0],[2.0,0.0]) ) == 3.0*2.0 );
}

private Area!Real[] splitArea(Real)( in Area!Real area, size_t dimension )
{
    assert( dimension < area.lower.length );
    auto div = (area.upper[dimension]-area.lower[dimension])
        //*0.5
        *uniform(0.4,0.6)
        + area.lower[dimension];
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
    assert( splitArea(a,0)[0].volume < a.volume );
    assert( splitArea(a,1)[0].volume < a.volume );
    assert( splitArea(a,0)[1].volume < a.volume );
    assert( splitArea(a,1)[1].volume < a.volume );

    auto splitted = splitArea(a,0);
    assert( a.volume == splitted[0].volume + splitted[1].volume );
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

///
Result!Real integrate(Func, Real)(scope Func f, Real[] a, Real[] b,
    Real epsRel = cast(Real) 1e-6, Real epsAbs = cast(Real) 0)
{
    auto area = Area!Real( a, b );
    auto result = miser(f, area, epsRel, epsAbs, 10*a.length);
    return Result!Real( result.value, 
            result.error ); 
}

///
Result!Real miser(Func, Real)(scope Func f, in Area!Real area,
    Real epsRel = cast(Real) 1e-6, Real epsAbs = cast(Real) 0, 
    size_t npoints = 1000 )
{
    assert( volume(area) > 0, "Size of area is 0" );
    auto bounds = area.lower.zip(area.upper);
    assert( bounds.all!((t) => t[1] > t[0] ) );
    auto points =
        iota( 0, npoints, 1 )
        .map!( (i) => bounds
                .map!( (t) {
                    return uniform!"[]"( t[0], t[1] ).to!Real; 
                    } 
                ).array 
             );
    auto values = points.map!((pnt) => f( pnt ) ).array;

    auto result = values.meanAndVariance(area);

    if (result.error < epsAbs 
            || result.error/result.value < epsRel)
        return result;

    // Try different subareas
    Area!Real[] bestAreas;
    auto bestEst = Real.max;
    Result!Real[] bestResults;
    foreach( j; 0..area.lower.length ) 
    {
        auto subAreas = area.splitArea( j );
        assert( volume(subAreas[0]) > 0, "Cannot divide the area further" );

        auto pntvs = zip( points, values );
        MeanSD[] msds = new MeanSD[2];
        foreach( pntv; zip( points, values ) )
        {
            if (pntv[0].withinArea(subAreas[0]))
                msds[0].put( pntv[1] );
            if (pntv[0].withinArea(subAreas[1]))
                msds[1].put( pntv[1] );
        }
        
        auto results = msds.zip(subAreas).map!((msd) => 
                meanAndVariance(msd[0], msd[1]) );
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

    auto sdA = bestResults[0].error;
    auto sdB = bestResults[1].error;
    auto sumSd = sdA + sdB;
    //assert( sumSd > 0, "Sum Errors to small" );
    if (sumSd == 0)
    {
        result = Result!Real( bestResults[0].value+bestResults[1].value, 
            sqrt( bestResults[0].error + bestResults[1].error ) );

        return result;
    }

    auto npntsl = round(150*area.dimension*sdA/sumSd).to!int; 
    auto npntsu = 150*area.dimension-npntsl;

    auto rl = miser( f, bestAreas[0], 
            epsRel, epsAbs,
            max( 5*area.dimension, npntsl ) );
    auto ru = miser( f, bestAreas[1], 
            epsRel, epsAbs,
            max( 5*area.dimension, npntsu ) );

    result = Result!Real( rl.value+ru.value, 
            sqrt(pow(rl.error,2)+pow(ru.error,2) ) );

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

    auto result = integrate( func, [-1.0,-1], [1.0,1.0], 1e-5, 0);
    assert( result.value <= PI + 1e-2 );
    assert( result.value >= PI - 1e-2 );
}
