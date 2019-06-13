
// The kernel file is "kernels_beam_elements_switchcaseoncpuremoved.cl"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <random>
#include <vector>
#include <iterator>
#include <fstream>
#include <sys/time.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h" // for NS(PATH_TO_BASE_DIR)
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/opencl/cl.h"

int main(int argc, char** argv)
{
      if(argc < 3) {
          std::cerr << "Usage: " << argv[0] << " < #particles > <#turns>  [deviceIdx]" << std::endl;
          exit(1);
        }
        double rtclock(void); // for timing the code
				double clkbegin, clkend; // for timing the code
    		double t; // for timing the code

  		int NUM_REPETITIONS = 10;; // for benchmarking
    	double num_of_turns = 0.0; // for timing
    	double average_execution_time = 0.0;

      std::vector<double> exec_time;

			for(int ll = 0; ll < NUM_REPETITIONS; ++ll) {
    /* We will use 9+ beam element blocks in this example and do not
     * care to be memory efficient yet; thus we make the blocks for
     * beam elements and particles big enough to avoid running into problems */

    constexpr st_block_size_t const MAX_NUM_BEAM_ELEMENTS       = 1000u; // 20u;
    constexpr st_block_size_t const NUM_OF_BEAM_ELEMENTS        = 1000u; //9u;

    /* 1MByte is plenty of space */
    constexpr st_block_size_t const BEAM_ELEMENTS_DATA_CAPACITY = 1048576u;

    /* Prepare and init the beam elements buffer */

    st_Blocks beam_elements;
    st_Blocks_preset( &beam_elements );

    int ret = st_Blocks_init( &beam_elements, MAX_NUM_BEAM_ELEMENTS,
                              BEAM_ELEMENTS_DATA_CAPACITY );

    (void)ret;
    assert( ret == 0 ); /* if there was an error, ret would be != 0 */

    /* Add NUM_OF_BEAM_ELEMENTS drifts to the buffer. For this example, let's
     * just have one simple constant length for all of them: */

   // One-fourth of the beam-elements are drift-elements
    for( st_block_size_t ii = 0 ; ii < NUM_OF_BEAM_ELEMENTS/4 ; ++ii )
    {
        double const drift_length = double{ 0.2L };
        st_Drift* drift = st_Blocks_add_drift( &beam_elements, drift_length );

        (void)drift; // using the variable with a no-op

        assert( drift != nullptr ); /* Otherwise, there was a problem! */
    }

    /* Check if we *really* have the correct number of beam elements and
     * if they really are all drifts */

    assert( st_Blocks_get_num_of_blocks( &beam_elements ) ==
            NUM_OF_BEAM_ELEMENTS/4 );

    /* The beam_elements container is currently not serialized yet ->
     * we could still add blocks to the buffer. Let's jus do this and
     * add a different kind of beam element to keep it easier apart! */

    for( st_block_size_t ii = NUM_OF_BEAM_ELEMENTS/4 ; ii < NUM_OF_BEAM_ELEMENTS/2 ; ++ii )
    {
        double const drift_length = double{ 0.1L };
    st_DriftExact* drift_exact = st_Blocks_add_drift_exact(
        &beam_elements, drift_length );
        (void) drift_exact;
    assert( drift_exact != nullptr );
   }

    assert( st_Blocks_get_num_of_blocks( &beam_elements ) ==
            ( NUM_OF_BEAM_ELEMENTS*0.5) );

    /* Adding the beam element 'cavity' */

    for( st_block_size_t ii = NUM_OF_BEAM_ELEMENTS*0.5 ; ii < NUM_OF_BEAM_ELEMENTS*0.75 ; ++ii )
    {
      double const voltage = double{ 1e4};
      double const frequency = double{ 40};
      double const lag = double{ 0.01L};
      st_Cavity* cavity = st_Blocks_add_cavity(
          &beam_elements, voltage, frequency, lag);
      (void) cavity; // a no-op
      assert( cavity != nullptr ); /* Otherwise, there was a problem! */
    }
    assert( st_Blocks_get_num_of_blocks( &beam_elements ) ==
            ( NUM_OF_BEAM_ELEMENTS * 0.75) );

    /* Adding the beam element 'align' */
    double const M__PI   = // note the two underscores between M and PI
      ( double )3.1415926535897932384626433832795028841971693993751L;
    for( st_block_size_t ii = NUM_OF_BEAM_ELEMENTS*0.75 ; ii < NUM_OF_BEAM_ELEMENTS ; ++ii )
    {
      double const tilt = double{ 0.5};
      double const z = double{ M__PI / 45};
      double const dx = double{ 0.2L};
      double const dy = double{ 0.2L};
      st_Align* align = st_Blocks_add_align(
          &beam_elements, tilt, cos( z ), sin( z ), dx, dy);
      (void) align; // a no-op
      assert( align != nullptr ); /* Otherwise, there was a problem! */
    }
    assert( st_Blocks_get_num_of_blocks( &beam_elements ) ==
        ( NUM_OF_BEAM_ELEMENTS) );
    /* Always safely terminate pointer variables pointing to resources they
     * do not own which we no longer need -> just a good practice */

//    drift_exact = nullptr;

    /* After serialization, the "structure" of the beam_elements buffer is
     * frozen, but the data in the elements - i.e. the length of the
     * individual drifts in our example - can still be modified. We will
     * just not be able to add further blocks to the container */

    assert( !st_Blocks_are_serialized( &beam_elements ) );

    ret = st_Blocks_serialize( &beam_elements );

    assert( ret == 0 );
    assert( st_Blocks_are_serialized( &beam_elements ) ); // serialization on CPU done.

    /* Next, let's iterate over all the beam_elements in the buffer and
     * print out the properties -> we expect that NUM_OF_BEAM_ELEMENTS
     * st_Drift with the same length appear and one st_DriftExact with a
     * different length should appear in the end */
    std::cout.flush();

/************************** Preparing grounds for OpenCL *******/
    std::vector<cl::Platform> platform;
    cl::Platform::get(&platform);

    if( platform.empty() )
    {
        std::cerr << "OpenCL platforms not found." << std::endl;
        return 1;
    }

    std::vector< cl::Device > devices;

    for( auto const& p : platform )
    {
        std::vector< cl::Device > temp_devices;

        p.getDevices( CL_DEVICE_TYPE_ALL, &temp_devices );

        for( auto const& d : temp_devices )
        {
            if( !d.getInfo< CL_DEVICE_AVAILABLE >() ) continue;
            devices.push_back( d );
        }
    }

    cl::Device* ptr_selected_device = nullptr;

    if( !devices.empty() )
    {
        if( argc >= 4 )
        {
            std::size_t const device_idx = std::atoi( argv[ 3 ] );

            if( device_idx < devices.size() )
            {
                ptr_selected_device = &devices[ device_idx ];
            }
        }

        if( ptr_selected_device == nullptr )
        {
            std::cout << "default selecting device #0" << std::endl;
            ptr_selected_device = &devices[ 0 ];
        }
    }

    if( ptr_selected_device != nullptr )
    {
        std::cout << "device: "
                  << ptr_selected_device->getInfo< CL_DEVICE_NAME >()
                  << std::endl;
    }
    else return 0;

    cl::Context context( *ptr_selected_device );

//    std::cout << "Device list" << std::endl;
//    for(unsigned int jj=0; jj<devices.size(); jj++){
//      std::cout << "Name of devicei " << jj<<" : "<<devices[jj].getInfo<CL_DEVICE_NAME>() << std::endl;
//      std::cout << "resolution of device timer for device " << jj <<" : "<<devices[jj].getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>() << std::endl;
//    };
/**********************************************/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // getting the kernel file
   std::string PATH_TO_KERNEL_FILE( st_PATH_TO_BASE_DIR );
       PATH_TO_KERNEL_FILE += "tests/benchmark/sixtracklib/opencl/";
       PATH_TO_KERNEL_FILE += "kernels_beam_elements_switchcaseoncpuremoved.cl";

       std::string kernel_source( "" );
       std::ifstream kernel_file( PATH_TO_KERNEL_FILE.c_str(),
                                  std::ios::in | std::ios::binary );

       if( kernel_file.is_open() )
       {
           std::istreambuf_iterator< char > file_begin( kernel_file.rdbuf() );
           std::istreambuf_iterator< char > end_of_file;

           kernel_source.assign( file_begin, end_of_file );
           kernel_file.close();
       }
////////////////////////////////////////////////////////////////////////////////////////////////////////////
    assert( ptr_selected_device != nullptr );

  //  int ndev = 0; // specifying the id of the device to be used
    cl::CommandQueue queue(context, *ptr_selected_device,CL_QUEUE_PROFILING_ENABLE);
    // Compile OpenCL program for found devices.
			cl:: Program program(context, kernel_source); //string  kernel_source contains the kernel(s) read from the file

#if 0
/////////////////////// Alternative 1 for including the kernels written in a separate file -- works perfectly fine /////////////////////////////////
			cl:: Program program(context, "#include \"../kernels.cl\" ", false); // the path inside the #include should be relative to an include directory specified using -Ipath/to/dir specified via build options.. otherwise give the absolute path.
#endif

#if 0
/////////////////////// The way to go if the string source[] contains the source in the same file as this.

//    cl::Program program(context, cl::Program::Sources(
//        1, std::make_pair(source, strlen(source))
//        ));
#endif


    try {
    std::string incls = "-D_GPUCODE=1 -D__NAMESPACE=st_ -I" + std::string(NS(PATH_TO_BASE_DIR));
  //  std::cout << "Path = " << incls << std::endl;
    //program.build(devices, "-D_GPUCODE=1 -D__NAMESPACE=st_ -I/home/sosingh/sixtracklib_gsoc18/initial_test/sixtrack-v0/external/include");
    program.build( incls.c_str() );
    } catch (const cl::Error&) {
    std::cerr
      << "OpenCL compilation error" << std::endl
      << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*ptr_selected_device)
      << std::endl;
    throw;
    }




    cl::Buffer B(context, CL_MEM_READ_WRITE, st_Blocks_get_total_num_bytes( &beam_elements  )); // input vector
queue.enqueueWriteBuffer( B, CL_TRUE, 0, st_Blocks_get_total_num_bytes( &beam_elements ), st_Blocks_get_const_data_begin( &beam_elements ) );


   ////////////////////////// Particles ////////////////////////////////
    st_block_size_t const NUM_PARTICLE_BLOCKS     = 1u;
    st_block_size_t const PARTICLES_DATA_CAPACITY = 1048576u*1000*4; //  ~(4 GB)
    st_block_size_t const NUM_PARTICLES           = atoi(argv[1]);

    st_Blocks particles_buffer;
    st_Blocks_preset( &particles_buffer );

    ret = st_Blocks_init(
        &particles_buffer, NUM_PARTICLE_BLOCKS, PARTICLES_DATA_CAPACITY );

    assert( ret == 0 );

    st_Particles* particles = st_Blocks_add_particles(
        &particles_buffer, NUM_PARTICLES );

    if( particles != nullptr )
    {
        /* Just some random values assigned to the individual attributes
         * of the acutal particles -> these values do not make any
         * sense physically, but should be safe for calculating maps ->
         * please check with the map for drift whether they do not produce
         * some NaN's at the sqrt or divisions by 0 though!*/

        std::mt19937_64  prng( 20180622 );

        std::uniform_real_distribution<> x_distribution(  0.05, 1.0 );
        std::uniform_real_distribution<> y_distribution(  0.05, 1.0 );
        std::uniform_real_distribution<> px_distribution( 0.05, 0.2 );
        std::uniform_real_distribution<> py_distribution( 0.05, 0.2 );
        std::uniform_real_distribution<> sigma_distribution( 0.01, 0.5 );

        assert( particles->s     != nullptr );
        assert( particles->x     != nullptr );
        assert( particles->y     != nullptr );
        assert( particles->px    != nullptr );
        assert( particles->py    != nullptr );
        assert( particles->sigma != nullptr );
        assert( particles->rpp   != nullptr );
        assert( particles->rvv   != nullptr );

        assert( particles->num_of_particles == (int)NUM_PARTICLES );

        for( st_block_size_t ii = 0 ; ii < NUM_PARTICLES ; ++ii )
        {
            particles->s[ ii ]     = 0.0;
            particles->x[ ii ]     = x_distribution( prng );
            particles->y[ ii ]     = y_distribution( prng );
            particles->px[ ii ]    = px_distribution( prng );
            particles->py[ ii ]    = py_distribution( prng );
            particles->sigma[ ii ] = sigma_distribution( prng );
            particles->rpp[ ii ]   = 1.0;
            particles->rvv[ ii ]   = 1.0;
        }
    }

    ret = st_Blocks_serialize( &particles_buffer );
    assert( ret == 0 );

    /* ===================================================================== */
    /* Copy to other buffer to simulate working on the GPU */
    //std::cout << "On the GPU:\n";

  // Allocate device buffers and transfer input data to device.

    cl::Buffer C(context, CL_MEM_READ_WRITE, st_Blocks_get_total_num_bytes( &particles_buffer )); // input vector
		queue.enqueueWriteBuffer( C, CL_TRUE, 0, st_Blocks_get_total_num_bytes( &particles_buffer ), st_Blocks_get_const_data_begin( &particles_buffer ) );

    int numThreads = 1;
    int blockSize = 1;
    cl::Kernel unserialize(program, "unserialize");
    unserialize.setArg(0,B);
    unserialize.setArg(1,C);
    unserialize.setArg(2,NUM_PARTICLES);
    queue.enqueueNDRangeKernel(
    unserialize, cl::NullRange, cl::NDRange( numThreads ),
    cl::NDRange(blockSize ));
    queue.flush();
    queue.finish();



      // creating a buffer to transfer the data from GPU to CPU

      std::vector< uint8_t > copy_particles_buffer_host(st_Blocks_get_total_num_bytes( &particles_buffer )/sizeof(uint8_t));  // output vector

      queue.enqueueReadBuffer(C, CL_TRUE, 0, copy_particles_buffer_host.size() * sizeof(uint8_t), copy_particles_buffer_host.data());
      queue.flush();

    st_Blocks copy_particles_buffer;
    st_Blocks_preset( &copy_particles_buffer );

    ret = st_Blocks_unserialize( &copy_particles_buffer, copy_particles_buffer_host.data() );
    assert( ret == 0 );


    SIXTRL_UINT64_T const NUM_TURNS = atoi(argv[2]); //100;

            cl::Kernel track_drift_particle(program, "track_drift_particle");
            blockSize = track_drift_particle.getWorkGroupInfo< CL_KERNEL_WORK_GROUP_SIZE >( *ptr_selected_device);// determine the work-group size
            numThreads = ((NUM_PARTICLES+blockSize-1)/blockSize) * blockSize; // rounding off NUM_PARTICLES to the next nearest multiple of blockSize. This is to ensure that there are integer number of work-groups launched
            std::cout << blockSize << " " << numThreads<< std::endl;
            track_drift_particle.setArg(0,B);
            track_drift_particle.setArg(1,C);
            //track_drift_particle.setArg(2,beam_index);
            track_drift_particle.setArg(3,NUM_PARTICLES);
            track_drift_particle.setArg(4,NUM_TURNS);


            cl::Kernel track_drift_exact_particle(program, "track_drift_exact_particle");
            blockSize = track_drift_exact_particle.getWorkGroupInfo< CL_KERNEL_WORK_GROUP_SIZE >( *ptr_selected_device);// determine the work-group size
            numThreads = ((NUM_PARTICLES+blockSize-1)/blockSize) * blockSize; // rounding off NUM_PARTICLES to the next nearest multiple of blockSize. This is to ensure that there are integer number of work-groups launched
            //std::cout << blockSize << " " << numThreads<< std::endl;
            track_drift_exact_particle.setArg(0,B);
            track_drift_exact_particle.setArg(1,C);
            //track_drift_exact_particle.setArg(2,beam_index);
            track_drift_exact_particle.setArg(3,NUM_PARTICLES);
            track_drift_exact_particle.setArg(4,NUM_TURNS);



            // enquing the cavity kernel
            cl::Kernel track_cavity_particle(program, "track_cavity_particle");
            blockSize = track_cavity_particle.getWorkGroupInfo< CL_KERNEL_WORK_GROUP_SIZE >( *ptr_selected_device);// determine the work-group size
            numThreads = ((NUM_PARTICLES+blockSize-1)/blockSize) * blockSize; // rounding off NUM_PARTICLES to the next nearest multiple of blockSize. This is to ensure that there are integer number of work-groups launched
           // std::cout << blockSize << " " << numThreads<< std::endl;
            track_cavity_particle.setArg(0,B);
            track_cavity_particle.setArg(1,C);
            //track_cavity_particle.setArg(2,beam_index);
            track_cavity_particle.setArg(3,NUM_PARTICLES);
            track_cavity_particle.setArg(4,NUM_TURNS);

            // enquing the align kernel
            cl::Kernel track_align_particle(program, "track_align_particle");
            blockSize = track_align_particle.getWorkGroupInfo< CL_KERNEL_WORK_GROUP_SIZE >( *ptr_selected_device);// determine the work-group size
            numThreads = ((NUM_PARTICLES+blockSize-1)/blockSize) * blockSize; // rounding off NUM_PARTICLES to the next nearest multiple of blockSize. This is to ensure that there are integer number of work-groups launched
           // std::cout << blockSize << " " << numThreads<< std::endl;
            track_align_particle.setArg(0,B);
            track_align_particle.setArg(1,C);
            //track_align_particle.setArg(2,beam_index);
            track_align_particle.setArg(3,NUM_PARTICLES);
            track_align_particle.setArg(4,NUM_TURNS);


#if 1
    // SIXTRL_UINT64_T beam_index = 500;
		clkbegin = rtclock();

    for(size_t nt=0; nt < NUM_TURNS; ++nt) {
    st_block_size_t beam_index = 0;

    /* Generate an iterator range over all the stored Blocks: */


    st_BlockInfo const* belem_it_begin =
      st_Blocks_get_const_block_infos_begin( &beam_elements );

    st_BlockInfo const* belem_end_finish =
      st_Blocks_get_const_block_infos_end( &beam_elements );

    cl::Event event;

    st_BlockInfo const* belem_it = belem_it_begin;
    st_BlockInfo const* belem_end = belem_it_begin + 250;
    for( ; belem_it != belem_end ; ++belem_it, ++beam_index )
    {
            track_drift_particle.setArg(2,beam_index);



            queue.enqueueNDRangeKernel(
                track_drift_particle, cl::NullRange, cl::NDRange( numThreads ),
                cl::NDRange(blockSize ), nullptr, &event);
            queue.flush();

     }

    belem_it = belem_it_begin + 250;
    belem_end = belem_it_begin + 500;
    for( ; belem_it != belem_end ; ++belem_it, ++beam_index )
    {

            track_drift_exact_particle.setArg(2,beam_index);

            queue.enqueueNDRangeKernel(
                track_drift_exact_particle, cl::NullRange, cl::NDRange( numThreads ),
                cl::NDRange(blockSize ), nullptr, &event);
            queue.flush();
        }

    belem_it = belem_it_begin + 500;
    belem_end = belem_it_begin + 750;
    for( ; belem_it != belem_end ; ++belem_it, ++beam_index )
    {
            track_cavity_particle.setArg(2,beam_index);

            queue.enqueueNDRangeKernel(
                track_cavity_particle, cl::NullRange, cl::NDRange( numThreads ),
                cl::NDRange(blockSize ), nullptr, &event);
            queue.flush();
     }

    belem_it = belem_it_begin + 750;
    belem_end = belem_end_finish;
    for( ; belem_it != belem_end ; ++belem_it, ++beam_index )
    {
            track_align_particle.setArg(2,beam_index);

            queue.enqueueNDRangeKernel(
                track_align_particle, cl::NullRange, cl::NDRange( numThreads ),
                cl::NDRange(blockSize ), nullptr, &event);
            queue.flush();
      }
    }// end of for loop for NUM_TURNS
	clkend = rtclock();
  t = clkend-clkbegin;
  exec_time.push_back(t);
        if( ll > 5 ) {
          num_of_turns += 1.0;
          average_execution_time += (t - average_execution_time)/num_of_turns;
      }
#endif



      queue.enqueueReadBuffer(C, CL_TRUE, 0, copy_particles_buffer_host.size() * sizeof(uint8_t), copy_particles_buffer_host.data());
      queue.flush();

    //st_Blocks copy_particles_buffer;
    st_Blocks_preset( &copy_particles_buffer );

    ret = st_Blocks_unserialize( &copy_particles_buffer, copy_particles_buffer_host.data() );
    assert( ret == 0 );

    /* on the GPU, these pointers will have __global as a decorator */

#if 0
    // On the CPU after copying the data back from the GPU
    std::cout << "\n On the Host, after applying the drift_track_particles mapping and copying from the GPU\n";

    SIXTRL_GLOBAL_DEC st_BlockInfo const* itr  =
        st_Blocks_get_const_block_infos_begin( &copy_particles_buffer );

    SIXTRL_GLOBAL_DEC st_BlockInfo const* endr =
        st_Blocks_get_const_block_infos_end( &copy_particles_buffer );

    for( ; itr != endr ; ++itr )
    {
        SIXTRL_GLOBAL_DEC st_Particles const* particles =
            ( SIXTRL_GLOBAL_DEC st_Particles const* )itr->begin;

        std::cout.precision( 4 );

        for( st_block_size_t ii = 0 ; ii < NUM_PARTICLES ; ++ii )
        {
            std::cout << " ii    = " << std::setw( 6 ) << ii
                      << std::fixed
                      << " | s     = " << std::setw( 6 ) << particles->s[ ii ]
                      << " | x     = " << std::setw( 6 ) << particles->x[ ii ]
                      << " | y     = " << std::setw( 6 ) << particles->y[ ii ]
                      << " | px    = " << std::setw( 6 ) << particles->px[ ii ]
                      << " | py    = " << std::setw( 6 ) << particles->py[ ii ]
                      << " | sigma = " << std::setw( 6 ) << particles->sigma[ ii ]
                      << " | rpp   = " << std::setw( 6 ) << particles->rpp[ ii ]
                      << " | rvv   = " << std::setw( 6 ) << particles->rvv[ ii ]
                      << "\r\n";
        }
    }

#endif
    std::cout.flush();
    st_Blocks_free( &particles_buffer );
    st_Blocks_free( &copy_particles_buffer );
  } // end of the NUM_REPETITIONS 'for' loop

    // printing the exec_time vector
    for(std::vector<double>::iterator it = exec_time.begin(); it != exec_time.end(); ++it)
      printf("%.3lf s%c",(*it), ",\n"[it+1 == exec_time.end()]);

		printf("Reference Version: Time = %.3lf s; \n",average_execution_time);
    return 0;

  }

double rtclock(void)
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

