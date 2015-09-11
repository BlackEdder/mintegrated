module mintegrated;

import std.array : array;
import std.conv : to;
import std.range : iota, zip;
import std.algorithm : all, map, max, reduce, filter;
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

/// Perform standard Monte Carlo integration
private MeanSD monteCarlo(Func, Real)( scope Func f, in Area!Real area, 
        in size_t npoints )
{
    auto bounds = area.lower.zip(area.upper);

    MeanSD msd;
    foreach( i; 0..npoints )
    {
        msd.put( f(
        bounds
                .map!( (t) {
                    return uniform!"[]"( t[0], t[1] ).to!Real; 
                    } 
                ).array ) );
    }
    return msd;
}

private Result!Real meanAndVariance(Real, MSD : MeanSD)( in MSD msd, in Area!Real area )
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
    auto result = miser(f, area, epsRel, epsAbs, 200000*a.length);
    return Result!Real( result.value, 
            result.error ); 
}

/// The returned error is the expected variance in the result
Result!Real miser(Func, Real)(scope Func f, in Area!Real area,
    Real epsRel = cast(Real) 1e-6, Real epsAbs = cast(Real) 0, 
    size_t npoints = 1000 )
{
    assert( volume(area) > 0, "Size of area is 0" );
    auto minPoints = 15*area.dimension;
    auto dim = max( 0.1*npoints, minPoints ).to!int;
    auto leftOverPoints = npoints - dim;

    if ( npoints < minPoints )
            //|| result.error < epsAbs 
        return monteCarlo(f, area, dim).meanAndVariance(area);

    // Try different subareas
    Area!Real[] bestAreas;
    auto bestEst = Real.max;
    Result!Real[] bestResults;
    foreach( j; 0..area.dimension ) 
    {
        auto subAreas = area.splitArea( j );
        assert( volume(subAreas[0]) > 0, "Cannot divide the area further" );

        MeanSD[] msds = new MeanSD[2];
        msds[0] = monteCarlo(f, subAreas[0], dim/2);
        msds[1] = monteCarlo(f, subAreas[1], dim/2);
        
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

    auto sdA = sqrt(bestResults[0].error);
    auto sdB = sqrt(bestResults[1].error);
    auto sumSd = sdA + sdB;
    //assert( sumSd > 0, "Sum Errors to small" );
    if (sumSd == 0)
    {
        auto result = Result!Real( bestResults[0].value+bestResults[1].value, 
            sqrt( bestResults[0].error + bestResults[1].error ) );

        return result;
    }

    auto npntsl = round(leftOverPoints*sdA/sumSd).to!int; 

    auto rl = miser( f, bestAreas[0], 
            epsRel, epsAbs, npntsl );
    auto ru = miser( f, bestAreas[1], 
            epsRel, epsAbs, leftOverPoints-npntsl );

    auto result = Result!Real( rl.value+ru.value, 
            rl.error+ru.error );

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

    auto result = integrate( func, [-1.0,-1], [1.0,1.0], 1e-5, 0 );
    result.writeln;
    assert( result.value <= PI + 3*sqrt(result.error) );
    assert( result.value >= PI - 3*sqrt(result.error) );
}

// Test precision
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

    MeanSD msd;

    foreach( i; 0..25 )
    {
        auto result = integrate( func, [-1.0,-1], [1.0,1.0], 1e-5, 0 );
        msd.put(result);
    }
    "Mean and variance".writeln;
    msd.mean.writeln;
    msd.var.writeln;
    assert( msd.var < 1e-4 );
}
///
unittest
{
    import std.stdio : writeln;
    auto func = function(double[] xs ) 
    {
        return xs[0]*xs[1];
    };
    auto result = integrate( func, [0.0,0], [1.0,1] );
    result.writeln;
    assert( result.value <= 0.25 + 3*sqrt(result.error) );
    assert( result.value >= 0.25 - 3*sqrt(result.error) );
}

///
unittest
{
    import std.math : PI, cos;
    import std.stdio : writeln;
    auto func = function(real[] xs ) 
    {
        return 1.0/(pow(PI,3)*(1-cos(xs[0])*cos(xs[1])*cos(xs[2])));
    };
    auto result = integrate( func, [0,0,0], [PI,PI,PI] );
    result.writeln;
    assert( result.value <= 1.393204 + 3*sqrt(result.error) );
    assert( result.value >= 1.393204 - 3*sqrt(result.error) );
}
