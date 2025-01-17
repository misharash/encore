#ifndef COMPUTE_MULTIPOLES_H
#define COMPUTE_MULTIPOLES_H

#ifdef GPU
#include "gpufuncs.h"
#endif

// ====================  Computing the pairs ==================

void compute_multipoles(Grid *grid, Float rmin, Float rmax) {
    int maxsep = ceil(rmax/grid->cellsize);   // Maximum distance we must search
    int ne;
    Float rmax2 = rmax*rmax;
    Float rmin2 = rmin*rmin; //rmax2*1e-12;    // Just an underflow guard
    uint64 cnt = 0;

    Multipoles *mlist = new Multipoles[MAXTHREAD*NBIN];  // Set up all of this space

    // Easy to multi-thread this top loop!
    // But some cells have trivial amounts of work, so we will first make a list of the work.
    // Including the empty cells appears to fool the dynamic thread allocation sometimes.

    STimer accmult, powertime; // measure the time spent accumulating powers for multipoles
    STimer sphtime;
    // We're going to loop only over the non-empty cells.

    long icnt = 0;
#ifdef GPU
    //maxp should not be more than ~1.5M
    int maxp = 1500000; //max number of particles - CHANGE ME TO CHANGE CHUNKING
    long dcnt = 0;
    int np = grid->np;
    int cellrange = 2*maxsep+1;
    int cells_per_particle = cellrange*cellrange*cellrange;
    //for PERIODIC or otherwise larger cellrange, ensure 600M is upper limit
    //for nmax
    if (600000000/cells_per_particle < maxp) maxp = 600000000/cells_per_particle;
    int pthresh = maxp/10;
    if (np < maxp) {
      maxp = np;
      pthresh = 0;
    }

    //nmax ~ cellrange^3*np - this will be number of threads run per kernel
    int nmax = (maxp+maxp/10)*cells_per_particle; //max size of arrays pnum, spnum, snp, sc
    int nthresh = nmax/10;
    int lastcell = 0;
    int lastcnt = 0;

    //allocate arrays for storing multipole data
    double *msave;
    int *csave;
    int *pnum, *spnum;
    int *snp, *sc;
    int *start_list, *np_list, *cellnums;

    //allocate arrays contianing pos and weights for each particle
    double *posx, *posy, *posz, *weights;

    //allocate arrays for adding pairs
    double *x0i, *x2i;

    if (_gpumode > 0) {

      if (_gpump == 1) {
        gpu_allocate_multipoles(&msave, &csave,
           &pnum, &spnum, &snp, &sc, NMULT, NBIN, maxp, nmax);
      } else if (_gpump == 2) {
        gpu_allocate_multipoles_fast(&msave, &csave,
		&start_list, &np_list, &cellnums,
		NMULT, NBIN, np, maxp, nmax, grid->ncells);
      }

      //allocate arrays contianing pos and weights for each particle
      gpu_allocate_particle_arrays(&posx, &posy, &posz, &weights, np);

      //allocate arrays for adding pairs
      gpu_allocate_pair_arrays(&x0i, &x2i, NBIN);

      if (_gpump == 2) {
        for (ne=0; ne<grid->nf; ne++) {
          int n = grid->filled[ne];  // Fetch the cell number
          Cell primary = grid->c[n];
          start_list[n] = primary.start;
          np_list[n] = primary.np;
          for (int j = primary.start; j<primary.start+primary.np; j++) {
            cellnums[j] = n;
          }
        }
      }

      //populate particle pos and weight
      for (int j = 0; j < np; j++) {
        weights[j] = grid->p[j].w;
        Float3 pos = grid->p[j].pos;
        posx[j] = pos.x;
        posy[j] = pos.y;
        posz[j] = pos.z;
      }

      //allocate alm arrays for up to maxp*NBIN*NLM size
      gpu_allocate_alms(maxp, NBIN, NLM, !_gpufloat && !_gpumixed);
    }

  #ifdef PERIODIC
    int *delta_x, *delta_y, *delta_z;
    if (_gpumode > 0) {
      gpu_allocate_periodic(&delta_x, &delta_y, &delta_z, nmax);
    }
  #endif
#endif

#ifdef OPENMP
#pragma omp parallel for schedule(dynamic,8) reduction(+:cnt)
#endif

    for (ne=0; ne<grid->nf; ne++) {
        int n = grid->filled[ne];  // Fetch the cell number

        // Decide which thread we are in
#ifdef OPENMP
	int thread = omp_get_thread_num();
        assert(omp_get_num_threads()<=MAXTHREAD);
        if (ne==0) printf("# Running on %d threads.\n", omp_get_num_threads());
#else
	int thread = 0;
        if (ne==0) printf("# Running single threaded.\n");
#endif
        if(int(ne%1000)==0) printf("Computing cell %d of %d on thread %d\n",ne,grid->nf,thread);
#ifdef FIVEPCF
        else if (int(ne%100)==0) printf("Computing cell %d of %d on thread %d\n",ne,grid->nf,thread);
#endif
    	// Loop over primary cells.
	Cell primary = grid->c[n];
	integer3 prim_id = grid->cell_id_from_1d(n);

        Multipoles *mult = mlist+thread*NBIN;   // Workspace for this thread

	// continue; // To skip all of the list-building and summations.
		// Everything else takes negligible time
	// Now we need to loop over all primary particles in this cell
	//For MP kernel 2 simply increment icnt.  Everything is done in kernel.
	if (_gpumode > 0 && _gpump == 2) icnt += primary.np; else
	for (int j = primary.start; j<primary.start+primary.np; j++) {
            int mloaded = 0;
	    if (smload && grid->p[j].w>=0) {
		// Start the multipoles from the input values
		// ONLY if the primary particle has weight>0
		int pid = grid->pid[j];
                for (int b=0; b<NBIN; b++) mult[b].load_and_reset(smload->fetchM(pid,b), smload->fetchC(pid,b));
		mloaded = 1;    // We'll use this to skip some pairs later.
	    } else {
		for (int b=0; b<NBIN; b++) mult[b].reset();   // Zero out the multipoles
	    }

	    Float primary_w = grid->p[j].w;
	    // Then loop over secondaries, cell-by-cell
	    integer3 delta;
            if(thread==0) accmult.Start();
    	    for (delta.x = -maxsep; delta.x <= maxsep; delta.x++)
	        for (delta.y = -maxsep; delta.y <= maxsep; delta.y++)
	            for (delta.z = -maxsep; delta.z <= maxsep; delta.z++) {
		        const int samecell = (delta.x==0&&delta.y==0&&delta.z==0)?1:0;

                        // Check that the cell is in the grid!
                        int tmp_test = grid->test_cell(prim_id+delta);
                        if(tmp_test<0) continue;
                        Cell sec = grid->c[tmp_test];

#ifdef GPU
			if (_gpumode > 0) {
			  //populate arrays only for GPU mode
			  //continue here, skip inner loop
                          pnum[dcnt] = j;
                          spnum[dcnt] = sec.start;
                          snp[dcnt] = sec.np;
                          sc[dcnt] = samecell;
  #ifdef PERIODIC
			  delta_x[dcnt] = (int)delta.x;
			  delta_y[dcnt] = (int)delta.y;
			  delta_z[dcnt] = (int)delta.z;
  #endif
                          dcnt++;
			  continue;
			}
#endif
                        // Define primary position
                        Float3 ppos = grid->p[j].pos;
#ifdef PERIODIC
                        ppos-=grid->cell_sep(delta);
#endif

		        // This is the position of the particle as viewed from the
		        // secondary cell.
		        // Now loop over the particles in this secondary cell
                        for (int k = sec.start; k<sec.start+sec.np; k++) {
		            // Now we're considering these two particles!
		            if (samecell&&j==k) continue;   // Exclude self-count
		            if (mloaded && grid->p[k].w>=0) continue;
		    	    // This particle has already been included in the file we loaded.
		            Float3 dx = grid->p[k].pos - ppos;
		            Float norm2 = dx.norm2();
		            // Check if this is in the correct binning ranges
                            if (norm2<rmax2 && norm2>rmin2) cnt++; else continue;

		            // Now what do we want to do with the pair?
		            norm2 = sqrt(norm2);  // Now just radius
		            // Find the radial bin
		            int bin = floor((norm2-rmin)/(rmax-rmin)*NBIN);

                            // Define x/r,y/r,z/r
		            dx = dx/norm2;

                            //continue;   // Skip pairs and multipoles

		            // Accumulate the 2-pt correlation function
		            // We include the weight for each pair
                            pairs[thread].add(bin, dx.z, grid->p[k].w*primary_w);
		            //continue;   // Skip the multipole creation

                            // Accumulate the multipoles

#ifdef AVX 	    // AVX only available for ORDER>=1
		            if (ORDER) mult[bin].addAVX(dx.x, dx.y, dx.z, grid->p[k].w);
			    else  mult[bin].add(dx.x, dx.y, dx.z, grid->p[k].w);
#else
		            mult[bin].add(dx.x, dx.y, dx.z, grid->p[k].w);
#endif
                        } // Done with this secondary particle
	            } // Done with this delta.z loop 
                //done with delta.y loop
            //done with delta.x loop
            for (int b=0; b<NBIN; b++) mult[b].finish();   // Finish the multipoles
            if(thread==0) accmult.Stop();

	    if (smsave && grid->p[j].w>=0) {
	        // We're saving multipoles, and this particle has positive weight.
		int pid = grid->pid[j];
		for (int b=0; b<NBIN; b++) mult[b].save(smsave->fetchM(pid,b), smsave->fetchC(pid,b));
	    }
	    // Now add these multipoles into the cross-powers
	    // This step takes very little time for the 3PCF, but is time-limiting for higher-point functions.
/*
	    //This code would save multipoles in CPU mode
	    for (int b = 0; b < NBIN; b++) {
	      mult[b].save(&msave[icnt*NBIN*NMULT+b*NMULT], &csave[icnt*NBIN+b]);
	    }
*/
	    icnt++;
	    if (_gpumode == 0) {
	      //This is done on CPU - calculate add_to_power here
	      //Acumulate powers here - code in NPCF.h (uses GPU kernels)
              if(thread==0) powertime.Start();
	      npcf[thread].add_to_power(mult, primary_w);
              if(thread==0) powertime.Stop();
#ifdef FIVEPCF
              if (int(ne%100)==0) printf("Powertime: %6.3f\n", powertime.Elapsed());
#endif
	    } 
	} // Done with this primary particle

#ifdef GPU
	//We are in last iteration of loop or are running into max size limits
	if (_gpumode > 0 && (dcnt > nmax-nthresh || icnt > maxp-pthresh || ne == (grid->nf)-1)) {
	  //needed for fast kernel
          int nthreads = cellrange*cellrange*cellrange*icnt;
	  if (_gpump == 1) nthreads = (int)dcnt;
	  cnt += nthreads;
	  bool usefloat = (_gpufloat || _gpumixed); 

          if(thread==0) accmult.Start();
  #ifdef PERIODIC
	  if (_only2pcf) {
	    printf("Running add_pairs_only_periodic kernel %d, mem mode %d, with %d threads after cell %d\n", _gpump, _shared, nthreads, ne);
            if (_gpump == 1) gpu_add_pairs_only_periodic(posx, posy, posz, weights, pnum, spnum, snp, sc, x0i, x2i, delta_x, delta_y, delta_z, (int)dcnt, NBIN, rmin, rmax, grid->cellsize, _shared, usefloat);
            else if (_gpump == 2) gpu_add_pairs_only_periodic_fast(posx, posy, posz, weights, start_list, np_list, cellnums, x0i, x2i, nthreads, NBIN, ORDER, NMULT, rmin, rmax, grid->nside_cuboid.x, grid->nside_cuboid.y, grid->nside_cuboid.z, grid->ncells, maxsep, lastcnt, grid->cellsize,_shared, usefloat);
	  } else {
	    printf("Running add_pairs_and_multipoles_periodic kernel %d, mem mode %d, with %d threads after cell %d\n", _gpump, _shared, nthreads, ne);
            if (_gpump == 1) gpu_add_pairs_and_multipoles_periodic(msave, posx, posy, posz, weights, csave, pnum, spnum, snp, sc, x0i, x2i, delta_x, delta_y, delta_z, (int)dcnt, NBIN, ORDER, NMULT, rmin, rmax, lastcnt, grid->cellsize, _shared, usefloat);
	    else if (_gpump == 2) gpu_add_pairs_and_multipoles_periodic_fast(msave, posx, posy, posz, weights, csave, start_list, np_list, cellnums, x0i, x2i, nthreads, NBIN, ORDER, NMULT, rmin, rmax, grid->nside_cuboid.x, grid->nside_cuboid.y, grid->nside_cuboid.z, grid->ncells, maxsep, lastcnt, grid->cellsize,_shared, usefloat);
	  }
  #else
	  if (_only2pcf) {
            printf("Running add_pairs_only kernel %d, mem mode %d, with %d threads after cell %d\n", _gpump, _shared, nthreads, ne); 
            if (_gpump == 1) gpu_add_pairs_only(posx, posy, posz, weights, pnum, spnum, snp, sc, x0i, x2i, (int)dcnt, NBIN, rmin, rmax, _shared, usefloat);
            else if (_gpump == 2) gpu_add_pairs_only_fast(posx, posy, posz, weights, start_list, np_list, cellnums, x0i, x2i, nthreads, NBIN, ORDER, NMULT, rmin, rmax, grid->nside_cuboid.x, grid->nside_cuboid.y, grid->nside_cuboid.z, grid->ncells, maxsep, lastcnt, _shared, usefloat);
	  } else {
            printf("Running add_pairs_and_multipoles kernel %d, mem mode %d, with %d threads after cell %d\n", _gpump, _shared, nthreads, ne);
	    if (_gpump == 1) gpu_add_pairs_and_multipoles(msave, posx, posy, posz, weights, csave, pnum, spnum, snp, sc, x0i, x2i, (int)dcnt, NBIN, ORDER, NMULT, rmin, rmax, lastcnt, _shared, usefloat); 
            else if (_gpump == 2) gpu_add_pairs_and_multipoles_fast(msave, posx, posy, posz, weights, csave, start_list, np_list, cellnums, x0i, x2i, nthreads, NBIN, ORDER, NMULT, rmin, rmax, grid->nside_cuboid.x, grid->nside_cuboid.y, grid->nside_cuboid.z, grid->ncells, maxsep, lastcnt, _shared, usefloat);
	  }
  #endif
          //gpu_device_synchronize(); //synchronize before copying data
          if(thread==0) accmult.Stop();

          if (_only2pcf) {
            gpu_device_synchronize(); //synchronize before copying data
            pairs[thread].load(x0i, x2i);
	    dcnt = 0;
            lastcell = ne+1;
            lastcnt += icnt;
	    icnt = 0;
	    continue;
	  }
	  //compute alms
          if(thread==0) sphtime.Start();
	  if (_gpufloat || _gpumixed) {
            gpu_compute_alms_float((int *)(npcf[0].map), msave, NBIN, NLM, maxp, ORDER, MAXORDER+1, NMULT);
	  } else {
	    gpu_compute_alms((int *)(npcf[0].map), msave, NBIN, NLM, maxp, ORDER, MAXORDER+1, NMULT);
	  }
          gpu_device_synchronize(); //synchronize before copying data
	  if(thread==0) sphtime.Stop();

          if(thread==0) powertime.Start();
          gpu_device_synchronize(); //synchronize before copying data
	  //call 3PCF add_to_power here
	  npcf[thread].add_to_power3_gpu(weights, icnt);
          //gpu_device_synchronize(); //synchronize before copying data

  #ifdef DISCONNECTED
          //call DISCONNECTED 4PCF here
	  npcf[thread].add_to_power_disconnected_gpu(weights, icnt);
  #endif

	  //reset dcnt and icnt
          dcnt = 0;
	  icnt = 0;
	  //copy 2PCF data here
	  pairs[thread].load(x0i, x2i);
	  //Loop over multipoles that have been calculated and call add_to_power
	  for (int nx = lastcell; nx <= ne; nx++) {
            int n = grid->filled[nx];  // Fetch the cell number

            if(int(nx%1000)==0) printf("Running add_to_power on cell %d of %d on thread %d\n",nx,grid->nf,thread);
            // Loop over primary cells.
            Cell primary = grid->c[n];

            Multipoles *mult = mlist+thread*NBIN;   // Workspace for this thread
            // Now we need to loop over all primary particles in this cell
            for (int j = primary.start; j<primary.start+primary.np; j++) {
                Float primary_w = grid->p[j].w;

	        //no longer need to copy back to Multipole object
		//since alm calcs are already done on GPU and GPU methods
		//exist for 3, 4, 5 PCF.
                //for (int b = 0; b < NBIN; b++) {
                //  mult[b].load_and_reset(&msave[icnt*NBIN*NMULT+b*NMULT], &csave[icnt*NBIN+b]);
                //}

                icnt++;
                npcf[thread].add_to_power(mult, primary_w);
            } // Done with this primary particle
          } // Done with this primary cell, end of omp pragma
	  lastcell = ne+1;
	  lastcnt += icnt;
	  if (ne != (grid->nf)-1) {
	    //need to reset msave and csave arrays to all zeros
	    //seems to be faster on CPU than GPU
	    for (int mx = 0; mx < maxp*NBIN*NMULT; mx++) msave[mx] = 0;
	    for (int cx = 0; cx < maxp*NBIN; cx++) csave[cx] = 0;
	  }
          if(thread==0) powertime.Stop();
	  icnt = 0;
	}
#endif
    } // Done with this primary cell, end of omp pragma

#ifdef GPU
    if (_gpumode > 0) {
        powertime.Start();
        gpu_device_synchronize(); //synchronize before copying data
        powertime.Stop();
        printf("\nGPU Spherical harmonics: %.3f s",sphtime.Elapsed());
        if (!_only2pcf) npcf[0].do_copy_memory();  //if not memcpy, must now copy back to host
        npcf[0].free_gpu_memory(); //free all GPU memory
	//need to free multipole / particle arrays too
	free_gpu_multipole_arrays(msave, csave, pnum, spnum, snp, sc, posx, posy, posz, weights, x0i, x2i);
  #ifdef PERIODIC
        free_gpu_periodic_arrays(delta_x, delta_y, delta_z);
  #endif
    }
#endif

  #ifdef GPUALM
    // free gpu memory
    gpu_free_mult(gmult,gmult_ct);
  #endif

#ifndef OPENMP
#ifdef AVX
    printf("\n# Time to compute required powers of x_hat, y_hat, z_hat (with AVX): %.2f\n\n",accmult.Elapsed());
#else
    printf("\n# Time to compute required powers of x_hat, y_hat, z_hat (no AVX): %.2f\n\n",accmult.Elapsed());
#endif
#endif

    printf("# We counted  %lld pairs within [%f %f].\n", cnt, rmin, rmax);
    printf("# Average of %f pairs per primary particle.\n",
    		(Float)cnt/grid->np);
    Float3 boxsize = grid->rect_boxsize;
    float expected = grid->np * (4*M_PI/3.0)*(pow(rmax,3.0)-pow(rmin,3.0))/(boxsize.x*boxsize.y*boxsize.z);
    printf("# We expected %1.0f pairs per primary particle, off by a factor of %f.\n", expected, cnt/(expected*grid->np));

    delete[] mlist;

    // Detailed timing breakdown
    printf("\n# Accumulate Powers: %6.3f s\n", accmult.Elapsed());
    printf("# Compute Power: %6.3f s\n\n", powertime.Elapsed());

    return;
}

#endif
