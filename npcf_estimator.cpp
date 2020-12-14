// npcf_estimator.cpp -- Oliver Philcox, 2020. Based on Daniel Eisenstein's 3PCF code.

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <complex>
#include <sys/time.h>
#include "threevector.hh"
#include "STimer.cc"

// For multi-threading:
#ifdef OPENMP
#include <omp.h>
#endif

// NBIN is the number of bins we'll sort the radii into.
#define NBIN 10

// ORDER is the order of the Ylm we'll compute.
// This must be <=MAXORDER, currently hard coded to 10.
#define ORDER 3

// MAXTHREAD is the maximum number of allowed threads.
// Big trouble if actual number exceeds this!
// No problem if actual number is smaller.
#define MAXTHREAD 40

typedef unsigned long long int uint64;

// Could swap between single and double precision here.
// Only double precision has been tested.
// Note that the AVX multipole code is always double precision.
typedef double Float;
typedef double3 Float3;
typedef std::complex<double> Complex;

// We need a vector floor3 function
Float3 floor3(float3 p) {
    return Float3(floor(p.x), floor(p.y), floor(p.z));
}

// we need a vector ceil3 function
Float3 ceil3(float3 p) {
    return Float3(ceil(p.x), ceil(p.y), ceil(p.z));
}

#define PAGE 4096     // To force some memory alignment.

// Classes specifying cells and grids
#include "modules/Basics.h"

// ========================== Accumulate the two-pcf pair counts ================

class Pairs {
  private:
    double *xi0, *xi2;

  private:
    double empty[8];   // Just to try to keep the threads from working on similar memory

  public:
    Pairs() {
	// Initialize the binning
	int ec=0;
    ec+=posix_memalign((void **) &xi0, PAGE, sizeof(double)*NBIN);
	ec+=posix_memalign((void **) &xi2, PAGE, sizeof(double)*NBIN);
    assert(ec==0);
	for (int j=0; j<NBIN; j++) {
	    xi0[j] = 0;
	    xi2[j] = 0;
	}
	empty[0] = 0.0;   // To avoid a warning
    }
    ~Pairs() {
        free(xi0);
        free(xi2);
    }

    inline void load(Float *xi0ptr, Float *xi2ptr) {
	for (int j=0; j<NBIN; j++) {
	    xi0[j] = xi0ptr[j];
	    xi2[j] = xi2ptr[j];
	}
    }

    inline void save(Float *xi0ptr, Float *xi2ptr) {
	for (int j=0; j<NBIN; j++) {
	    xi0ptr[j] = xi0[j];
	    xi2ptr[j] = xi2[j];
	}
    }

    inline void add(int b, Float dz, Float w) {
	// Add up the weighted pair for the monopole and quadrupole correlation function
        xi0[b] += w;
        xi2[b] += w*(3.0*dz*dz-1)*0.5;
    }

    void sum_power(Pairs *p) {
	// Just add up all of the threaded pairs into the zeroth element
	for (int i=0; i<NBIN; i++) {
        xi0[i] += p->xi0[i];
	    xi2[i] += p->xi2[i];
	}
    }

    void report_pairs() {
    for (int j=0; j<NBIN; j++) {
	    printf("Pairs %2d %9.0f %9.0f\n",
			j, xi0[j], xi2[j]);
	}
    }
};

Pairs pairs[MAXTHREAD];

// ====================  Setting up the multipoles ==================

// The total number of Cartesian multipoles, satisfying a+b+c<=ORDER
#define NMULT ((ORDER+1)*(ORDER+2)*(ORDER+3)/6)
// We adopt a convention in which we loop over ell and then m=0..ell
#define NLM ((ORDER+1)*(ORDER+2)/2)

#ifdef AVX
#include "externalmultipoles.h"
// typedef struct { double v[4]; } d4;

// An array of pointers to all of the AVX assembly functions
void (*CMptr[16])( d4 *ip1x, d4 *ip2x, d4 *ip1y, d4 *ip2y, d4 *ip1z, d4 *ip2z,
                   d4 *cx, d4 *cy, d4 *cz, d4 *globalM,
                   d4 *mass1, d4 *mass2) = {
     MultipoleKernel1,  MultipoleKernel2,  MultipoleKernel3,  MultipoleKernel4,
     MultipoleKernel5,  MultipoleKernel6,  MultipoleKernel7,  MultipoleKernel8,
     MultipoleKernel9,  MultipoleKernel10, MultipoleKernel11, MultipoleKernel12,
     MultipoleKernel13, MultipoleKernel14, MultipoleKernel15, MultipoleKernel16
};

#endif

// Here's a simple structure for our normalized differences of the positions
typedef struct Xdiff {
    Float dx, dy, dz, w;
} Xdiff;

// Include multipoles class
#include "modules/Multipoles.h"

// Include multipole storage code
#include "modules/StoreMultipoles.h"

StoreMultipoles *smload, *smsave;

// Include the NPCF class here
#include "modules/NPCF.h"

NPCF npcf[MAXTHREAD];

void zero_power() {
    for (int t=0; t<MAXTHREAD; t++) npcf[t].reset();
}

void sum_power() {
    // Just add up all of the threaded power into the zeroth element
    for (int t=0; t<MAXTHREAD; t++)
	printf("# Bin 0 counter for thread %2d: %9lld\n", t, npcf[t].bincounts[0]);
    for (int t=1; t<MAXTHREAD; t++)
        npcf[0].sum_power(npcf+t);
    for (int t=1; t<MAXTHREAD; t++)
        pairs[0].sum_power(pairs+t);
    return;
}

// Include class which creates the multipoles, including the special AVX coding
#include "modules/ComputeMultipoles.h"

// Include class which creates / reads in particles and assigns them to a grid
#include "modules/Driver.h"

// ================================ main() =============================

void usage() {
    fprintf(stderr, "\nUsage for grid_multipoles/grid_multipolesAVX:\n");
    fprintf(stderr, "   -box <boxsize> : The periodic size of the computational domain.  Default 400.\n");
    fprintf(stderr, "   -scale <rescale>: How much to dilate the input positions by.  Default 0.\n");
    fprintf(stderr, "             Zero or negative value causes =boxsize, rescaling unit cube to full periodicity\n");
    fprintf(stderr, "   -rmax <rmax>: The maximum radius of the largest pair bin.  Default 200.\n");
    fprintf(stderr, "   -nside <nside>: The grid size for accelerating the pair count.  Default 8.\n");
    fprintf(stderr, "             Recommend having several grid cells per rmax.\n");
    fprintf(stderr, "   -in <file>: The input file (space-separated x,y,z,w).  Default sample.dat.\n");
    fprintf(stderr, "   -outstr <outstring>: String to prepend to the output file.  Default sample.\n");
    fprintf(stderr, "   -ran <np>: Ignore any file and use np random perioidic points instead.\n");
    fprintf(stderr, "   -def: This allows one to accept the defaults without giving other entries.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Two other important parameters can only be set during compilations:\n");
    fprintf(stderr, "   ORDER: The multipole order being computed.\n");
    fprintf(stderr, "   NBIN:  The number of radial bins.\n");
    fprintf(stderr, "Similarly, the radial bin spacing (currently linear) is hard-coded.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "For advanced use, there is an option store the multipoles of positively weighted primary particles.\n");
    fprintf(stderr, "    -save <filename>: Triggers option to store the multipoles.\n");
    fprintf(stderr, "The file can then be reloaded on subsequent runes\n");
    fprintf(stderr, "    -load <filename>: Triggers option to load the multipoles\n");
    fprintf(stderr, "The intention is to allow re-use of DD counts while changing the DR and RR counts.\n");
    fprintf(stderr, "    -balance: Rescale the negative weights so that the total weight is zero.\n");
    fprintf(stderr, "    -invert: Multiply all the weights by -1.\n");

    exit(1);
    return;
}

int main(int argc, char *argv[]) {
    // Important variables to set!  Here are the defaults:
    Float boxsize = 1;
        // The periodicity of the position-space cube. (overwritten if reading from file)
    Float rescale = 0.0;   // If left zero or negative, set rescale=boxsize
    	// The particles will be read from the unit cube, but then scaled by boxsize.
    Float rmax = 0.05;
    	// The maximum radius of the largest bin.
    int nside = 20;
	// The grid size, which should be tuned to match boxsize and rmax.
        // Don't forget to adjust this if changing boxsize!
    int make_random = 0;
    	// If set, we'll just throw random periodic points instead of reading the file
    int np = -1;   // Will be number of particles in a random distribution,
    	// but gets overwritten if reading from a file.
    int qbalance = 0, qinvert = 0;
    const char default_fname[] = "sample.dat";
    const char default_outstr[] = "sample";
    char *fname = NULL;
    char *outstr = NULL;
    char *savename = NULL;
    char *loadname = NULL;

    // The periodicity of the position-space cuboid in 3D.
    Float3 rect_boxsize = {boxsize,boxsize,boxsize}; // this is overwritten on particle read-in
    Float cellsize;

    STimer TotalTime, Prologue, Epilogue, MultipoleTime, IOTime;

    TotalTime.Start();
    Prologue.Start();
    if (argc==1) usage();
    int i=1;
    while (i<argc) {
	     if (!strcmp(argv[i],"-boxsize")||!strcmp(argv[i],"-box")){
             Float tmp_box = atof(argv[++i]);
             rect_boxsize={tmp_box,tmp_box,tmp_box};
         }
	else if (!strcmp(argv[i],"-rescale")||!strcmp(argv[i],"-scale")) rescale = atof(argv[++i]);
	else if (!strcmp(argv[i],"-rmax")||!strcmp(argv[i],"-max")) rmax = atof(argv[++i]);
	else if (!strcmp(argv[i],"-nside")||!strcmp(argv[i],"-ngrid")||!strcmp(argv[i],"-grid")) nside = atoi(argv[++i]);
  else if (!strcmp(argv[i],"-in")) fname = argv[++i];
  else if (!strcmp(argv[i],"-outstr")) outstr = argv[++i];
	else if (!strcmp(argv[i],"-save")||!strcmp(argv[i],"-store")) savename = argv[++i];
	else if (!strcmp(argv[i],"-load")) loadname = argv[++i];
	else if (!strcmp(argv[i],"-balance")) qbalance = 1;
	else if (!strcmp(argv[i],"-invert")) qinvert = 1;
	else if (!strcmp(argv[i],"-ran")||!strcmp(argv[i],"-np")) {
		double tmp;
		if (sscanf(argv[++i],"%lf", &tmp)!=1) {
		    fprintf(stderr, "Failed to read number in %s %s\n",
		    	argv[i-1], argv[i]);
		    usage();
		}
		np = tmp;
		make_random=1;
	    }
	else if (!strcmp(argv[i],"-def")||!strcmp(argv[i],"-default")) { fname = NULL; }
	else {
	    fprintf(stderr, "Don't recognize %s\n", argv[i]);
	    usage();
	}
	i++;
    }

    // Compute smallest and largest boxsizes
    Float box_min = fmin(fmin(rect_boxsize.x,rect_boxsize.y),rect_boxsize.z);
    Float box_max = fmax(fmax(rect_boxsize.x,rect_boxsize.y),rect_boxsize.z);

    assert(i==argc);  // For example, we might have omitted the last argument, causing disaster.

    assert(box_min>0.0);
    assert(rmax>0.0);
    assert(nside>0);
    assert(nside<300);   // Legal, but rather unlikely that we should use something this big!
    if (rescale<=0.0) rescale = box_max;   // This would allow a unit cube to fill the periodic volume
    if (fname==NULL) fname = (char *) default_fname;   // No name was given
    if (outstr==NULL) outstr = (char *) default_outstr;   // No outstring was given

    // Output for posterity
    printf("Box Size = {%6.5e,%6.5e,%6.5e}\n", rect_boxsize.x,rect_boxsize.y,rect_boxsize.z);
    printf("Grid = %d\n", nside);
    printf("Maximum Radius = %6.1g\n", rmax);
    Float gridsize = rmax/(box_max/nside);
    printf("Radius in Grid Units = %6.3g\n", gridsize);
    if (gridsize<1) printf("#\n# WARNING: grid appears inefficiently coarse\n#\n");
    printf("Bins = %d\n", NBIN);
    printf("Order = %d\n", ORDER);
    assert(ORDER<=MAXORDER);   // Actually, this will run, but it would give silent zeros.

    Particle *orig_p;
    Float3 shift;
    if (make_random) {
        // If you want to just make random particles instead:
        assert(np>0);
        assert(boxsize==rescale);    // Nonsense if not!
        orig_p = make_particles(rect_boxsize, np);
        cellsize = rect_boxsize.x/nside; // define size of cells
    } else {
        orig_p = read_particles(rescale, &np, fname);
        assert(np>0);
        // Update boxsize here
        compute_bounding_box(orig_p,np,rect_boxsize,cellsize,rmax,shift,nside);
    }

    if (qinvert) invert_weights(orig_p, np);
    if (qbalance) balance_weights(orig_p, np);

    // Compute the NPCF weights using the array of (squared) a_lm normalizations
    //TODO: maybe precompute this?
    // Now include the additional factor of (-1)^l / Sqrt(2l+1) and *2 for m != 0 mode (from m -> -m symmetry)
    for(int ell=0, n=0; ell<=ORDER; ell++){
      for(int m=0; m<=ell; m++, n++){
          weight3pcf[n] = almnorm[n]*pow(-1.,ell)/sqrt(2.*ell+1.);
          if(m!=0) weight3pcf[n] *= 2.0;
        }
    }

    // Now ready to compute!
    // Sort the particles into the grid.
    Grid grid(orig_p, np, rect_boxsize, cellsize,shift);
    printf("# Done gridding the particles\n");
    printf("# %d particles in use, %d with positive weight\n", grid.np, grid.np_pos);
    printf("# Weights: Positive particles sum to %f\n", grid.sumw_pos);
    printf("#          Negative particles sum to %f\n", grid.sumw_neg);
    free(orig_p);

    Float grid_density = (double)np/grid.nf;
    printf("Average number of particles per grid cell = %6.2g\n", grid_density);
    printf("Average number of particles per max_radius ball = %6.2g\n",
	np*4.0*M_PI/3.0*pow(rmax,3.0)/(rect_boxsize.x*rect_boxsize.y*rect_boxsize.z));
    if (grid_density<1) printf("#\n# WARNING: grid appears inefficiently fine.\n#\n");

    smsave = smload = NULL;
    if (loadname!=NULL) smload = new StoreMultipoles(grid.np_pos);
    if (savename!=NULL) smsave = new StoreMultipoles(grid.np_pos);

    IOTime.Start();
    if (smload!=NULL) {
        smload->load(loadname);
	pairs[0].load(smload->xi0, smload->xi2);
	    // Put all of the previous work in thread 0
    }
    IOTime.Stop();

    zero_power();
    fflush(NULL);
    Prologue.Stop();

    // Everything above here takes negligible time.  This line is nearly all of the work.
    MultipoleTime.Start();
    compute_multipoles(&grid, rmax);
    printf("# Done counting the pairs\n");
    MultipoleTime.Stop();

    // Output the results
    Epilogue.Start();
    sum_power();
    printf("\n# Binned weighted pair counts, monopole and quadrupole\n");
    pairs[0].report_pairs();

    printf("\n# Multipole power\n");
    npcf[0].report_power();

    // Save the outputs
    npcf[0].save_power(outstr, rmax);

    IOTime.Start();
    if (smsave!=NULL) {
	pairs[0].save(smsave->xi0, smsave->xi2);
        smsave->save(savename);
    }
    IOTime.Stop();

    if (smload!=NULL) delete smload;
    if (smsave!=NULL) delete smsave;
    Epilogue.Stop();
    TotalTime.Stop();
    printf("\n# Total Time: %4.1f s\n", TotalTime.Elapsed());
    printf("# Prologue: %6.3f s (%4.1f%%)\n", Prologue.Elapsed(), Prologue.Elapsed()/TotalTime.Elapsed()*100.0);
    printf("# Epilogue: %6.3f s (%4.1f%%)\n", Epilogue.Elapsed(), Epilogue.Elapsed()/TotalTime.Elapsed()*100.0);
    printf("# IO Time:  %6.3f s (%4.1f%%)\n", IOTime.Elapsed(), IOTime.Elapsed()/TotalTime.Elapsed()*100.0);
    printf("# Pairs:    %6.3f s (%4.1f%%)\n", MultipoleTime.Elapsed(), MultipoleTime.Elapsed()/TotalTime.Elapsed()*100.0);
    return 0;
}
