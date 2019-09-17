#include "tests/benchmark/toml.h"
#include "sixtracklib/sixtracklib.hpp"

#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace SIXTRL_CXX_NAMESPACE
{
    namespace benchmark
    {
        struct MainConfig
        {
            MainConfig( char const* path_to_config_file = nullptr ) :
                config( nullptr ), git_branch_str( nullptr ),
                git_hash_str( nullptr ), name_str( nullptr ),
                output_path( nullptr )
            {
                if( path_to_config_file != nullptr )
                {
                    bool const success = this->init( path_to_config_file );
                    SIXTRL_ASSERT( success );
                    ( void )success;
                }
            }

            MainConfig( MainConfig const& ) = delete;
            MainConfig( MainConfig&& ) = default;

            MainConfig& operator=( MainConfig const& ) = delete;
            MainConfig& operator=( MainConfig&& ) = default;

            ~MainConfig()
            {
                if( this->config != nullptr ) ::toml_free( this->config );
                this->clear();
            }

            bool init( char const* path_to_config_file )
            {
                if( path_to_config_file == nullptr ) return false;

                if( this->config != nullptr )
                {
                    ::toml_free( this->config );
                }

                bool success = false;
                this->clear();

                ::FILE* fp = std::fopen( path_to_config_file, "r" );

                if( fp != nullptr )
                {
                    std::vector< char > error_msg( 256, '\0' );
                    this->config = ::toml_parse_file(
                        fp, error_msg.data(), error_msg.size() );

                    std::fclose( fp );
                    fp = nullptr;

                    if( this->config != nullptr )
                    {
                        success = true;

                        char const* raw_str =
                            ::toml_raw_in( this->config, "git_hash" );

                        if( raw_str != nullptr )
                        {
                            if( 0 != ::toml_rtos( raw_str, &this->git_hash_str ) )
                            {
                                success = true;
                            }
                        }

                        raw_str = ::toml_raw_in( this->config, "git_branch" );

                        if( raw_str != nullptr )
                        {
                            if( 0 != ::toml_rtos( raw_str, &this->git_branch_str ) )
                            {
                                success = false;
                            }
                        }

                        raw_str = ::toml_raw_in( this->config, "name" );

                        if( raw_str != nullptr )
                        {
                            if( 0 != ::toml_rtos( raw_str, &this->name_str ) )
                            {
                                success = false;
                            }
                        }
                        else
                        {
                            success = false;
                        }

                        raw_str = ::toml_raw_in( this->config, "output_path" );

                        if( raw_str != nullptr )
                        {
                            if( 0 != ::toml_rtos( raw_str, &this->output_path ) )
                            {
                                success = false;
                            }
                        }
                    }
                    else
                    {
                        std::ostringstream a2str;
                        a2str << "Unable to parse config file\r\n"
                              << "Config File: " << path_to_config_file
                              << "\r\n"
                              << "Error      : " << error_msg.data();

                        std::cerr << a2str.str() << std::endl;
                    }
                }

                return success;
            }

            void clear()
            {
                if( this->git_hash_str != nullptr )
                {
                    free( this->git_hash_str );
                    this->git_hash_str = nullptr;
                }

                if( this->git_branch_str != nullptr )
                {
                    free( this->git_branch_str );
                    this->git_branch_str = nullptr;
                }

                if( this->name_str != nullptr )
                {
                    free( this->name_str );
                    this->name_str = nullptr;
                }

                if( this->output_path != nullptr )
                {
                    free( this->output_path );
                    this->output_path = nullptr;
                }
            }

            ::toml_table_t* config;

            char* git_branch_str;
            char* git_hash_str;
            char* name_str;
            char* output_path;
        };

        struct TargetConfig
        {
            TargetConfig() : target_conf( nullptr ), arch_str( nullptr ),
                node_id_str( nullptr ), config_str( nullptr )
            {

            }

            TargetConfig( MainConfig& main ) :
                target_conf( nullptr ), arch_str( nullptr ),
                node_id_str( nullptr ), config_str( nullptr )
            {
                bool const success = this->init( main );
                SIXTRL_ASSERT( success );
                ( void )success;
            }

            TargetConfig( TargetConfig const& ) = default;
            TargetConfig( TargetConfig&& ) = default;

            TargetConfig& operator=( TargetConfig const& ) = default;
            TargetConfig& operator=( TargetConfig&& ) = default;

            ~TargetConfig()
            {
                this->clear();
            }

            bool clear()
            {
                this->target_conf = nullptr;

                if( this->arch_str != nullptr )
                {
                    free( this->arch_str );
                    this->arch_str = nullptr;
                }

                if( this->node_id_str != nullptr )
                {
                    free( this->node_id_str );
                    this->node_id_str = nullptr;
                }

                if( this->config_str != nullptr )
                {
                    free( this->config_str  );
                    this->config_str = nullptr;
                }
            }

            bool init( MainConfig& main )
            {
                bool success = false;
                char const* raw_str = nullptr;

                if( this->target_conf != nullptr ) return false;
                if( main.config == nullptr ) return false;

                this->clear();

                this->target_conf = ::toml_table_in( main.config, "target" );

                if( this->target_conf != nullptr )
                {
                    success = true;
                    raw_str = ::toml_raw_in( this->target_conf, "arch" );

                    if( raw_str != nullptr )
                    {
                        if( 0 != ::toml_rtos( raw_str, &this->arch_str ) )
                        {
                            success = false;
                        }
                    }

                    raw_str = ::toml_raw_in( this->target_conf, "node_id" );

                    if( raw_str != nullptr )
                    {
                        if( 0 != ::toml_rtos( raw_str, &this->node_id_str ) )
                        {
                            success = false;
                        }
                    }

                    raw_str = ::toml_raw_in( this->target_conf, "config_str" );

                    if( raw_str != nullptr )
                    {
                        if( 0 != ::toml_rtos( raw_str, &this->config_str ) )
                        {
                            success = false;
                        }
                    }
                }

                if( success )
                {
                    if( ( arch_str != nullptr ) &&
                        ( ( std::strcmp( arch_str, "opencl" ) == 0 ) ||
                          ( std::strcmp( arch_str, "cuda" ) == 0 ) ) &&
                        ( node_id_str == nullptr ) )
                    {
                        success = false;
                    }
                }

                return success;
            }

            ::toml_table_t* target_conf;

            char* arch_str;
            char* node_id_str;
            char* config_str;
        };

        struct TrackItem
        {
            TrackItem() : num_turns( 0 ), num_particles( 0 ),
                num_repetitions( 0 ), be_start_idx( -1 ), be_stop_idx( -1 ),
                finish_turn( false ), elem_by_elem( false )
            {

            }

            TrackItem( TrackItem const& ) = default;
            TrackItem( TrackItem&& ) = default;
            TrackItem& operator=( TrackItem const& ) = default;
            TrackItem& operator=( TrackItem&& ) = default;
            ~TrackItem() = default;

            bool isTrackUntilItem() const
            {
                return ( ( this->num_turns > uint64_t{ 0 } ) &&
                         ( this->num_repetitions > uint64_t{ 0 } ) &&
                         ( this->num_particles > uint64_t{ 0 } ) &&
                         ( !this->elem_by_elem ) );
            }

            bool isTrackElemByElemItem() const
            {
                return ( ( this->num_turns > uint64_t{ 0 } ) &&
                         ( this->num_particles > uint64_t{ 0 } ) &&
                         ( this->elem_by_elem ) );
            }

            bool isTrackLineItem() const
            {
                return ( ( this->num_particles > uint64_t{ 0 } ) &&
                         ( this->be_start_idx >= int64_t{ 0 } ) &&
                         ( this->be_stop_idx  >= this->be_start_idx ) &&
                         ( !this->elem_by_elem ) );
            }

            uint64_t num_turns;
            uint64_t num_particles;
            uint64_t num_repetitions;

            int64_t be_start_idx;
            int64_t be_stop_idx;
            bool     finish_turn;
            bool     elem_by_elem;
        };

        struct TrackConfig
        {
            TrackConfig() : track_items(), track_config( nullptr ),
                path_particle_dump( nullptr ),  path_lattice_dump( nullptr )
            {

            }

            TrackConfig( MainConfig& main ) :
                track_items(), track_config( nullptr ),
                path_particle_dump( nullptr ),  path_lattice_dump( nullptr )
            {
                bool const success = this->init( main );
                SIXTRL_ASSERT( success );
                ( void )success;
            }

            TrackConfig( TrackConfig const& ) = default;
            TrackConfig( TrackConfig&& ) = default;
            TrackConfig& operator=( TrackConfig const& ) = default;
            TrackConfig& operator=( TrackConfig&& ) = default;

            ~TrackConfig()
            {
                this->clear();
            }

            bool init( MainConfig& main )
            {
                bool success = false;
                char const* raw_str = nullptr;

                if( this->track_config != nullptr ) return false;
                if( main.config == nullptr ) return false;

                this->clear();
                this->track_config = ::toml_table_in( main.config, "track" );

                if( this->track_config != nullptr )
                {
                    success = true;
                    raw_str = ::toml_raw_in(
                        this->track_config, "path_particle_dump" );

                    if( raw_str != nullptr )
                    {
                        if( 0 != ::toml_rtos( raw_str,
                                &this->path_particle_dump ) )
                        {
                            success = false;
                        }
                    }
                    else
                    {
                        success = false;
                    }

                    raw_str = ::toml_raw_in(
                        this->track_config, "path_lattice_dump" );

                    if( raw_str != nullptr )
                    {
                        if( 0 != ::toml_rtos(
                                raw_str, &this->path_lattice_dump ) )
                        {
                            success = false;
                        }
                    }
                    else
                    {
                        success = false;
                    }

                    if( success )
                    {
                        ::toml_array_t* ptr_num_particles = ::toml_array_in(
                            this->track_config, "num_particles" );

                        ::toml_array_t* ptr_num_turns = ::toml_array_in(
                            this->track_config, "num_turns" );

                        ::toml_array_t* ptr_num_repetitions = ::toml_array_in(
                            this->track_config, "num_repetitions" );

                        if( ( ptr_num_particles != nullptr ) &&
                            ( ::toml_array_kind( ptr_num_particles ) == 'v' ) &&
                            ( ::toml_array_type( ptr_num_particles ) == 'i' ) &&
                            ( ::toml_array_nelem( ptr_num_particles ) > 0 ) &&
                            ( ptr_num_turns != nullptr ) &&
                            ( ::toml_array_kind( ptr_num_turns ) == 'v' ) &&
                            ( ::toml_array_type( ptr_num_turns ) == 'i' ) &&
                            ( ::toml_array_nelem( ptr_num_turns ) ==
                              ::toml_array_nelem( ptr_num_particles ) ) &&
                            ( ptr_num_repetitions != nullptr ) &&
                            ( ::toml_array_kind( ptr_num_repetitions ) == 'v' ) &&
                            ( ::toml_array_type( ptr_num_repetitions ) == 'i' ) &&
                            ( ::toml_array_nelem( ptr_num_repetitions ) ==
                              ::toml_array_nelem( ptr_num_particles ) ) )
                        {
                            int const nn =
                                ::toml_array_nelem( ptr_num_particles );

                            for( int ii = 0 ; ii < nn ; ++ii )
                            {
                                int64_t temp_num_particles   = int64_t{ 0 };
                                int64_t temp_num_turns       = int64_t{ 0 };
                                int64_t temp_num_repetitions = int64_t{ 0 };

                                char const* raw_str = ::toml_raw_at(
                                    ptr_num_particles, ii );

                                if( raw_str != nullptr )
                                {
                                    if( 0 != ::toml_rtoi(
                                            raw_str, &temp_num_particles ) )
                                    {
                                        success = false;
                                    }
                                }
                                else
                                {
                                    success = false;
                                }

                                raw_str = ::toml_raw_at( ptr_num_turns, ii );

                                if( raw_str != nullptr )
                                {
                                    if( 0 != ::toml_rtoi(
                                            raw_str, &temp_num_turns ) )
                                    {
                                        success = false;
                                    }
                                }
                                else
                                {
                                    success = false;
                                }

                                raw_str = ::toml_raw_at(
                                    ptr_num_repetitions, ii );

                                if( raw_str != nullptr )
                                {
                                    if( 0 != ::toml_rtoi(
                                            raw_str, &temp_num_repetitions ) )
                                    {
                                        success = false;
                                    }
                                }
                                else
                                {
                                    success = false;
                                }

                                if( ( temp_num_particles < int64_t{ 0 } ) ||
                                    ( temp_num_turns < int64_t{ 0 } ) ||
                                    ( temp_num_repetitions < int64_t{ 0 } ) )
                                {
                                    success = false;
                                }

                                if( success )
                                {
                                    this->track_items.push_back( TrackItem{} );

                                    this->track_items.back().num_particles =
                                        temp_num_particles;

                                    this->track_items.back().num_turns =
                                        temp_num_turns;

                                    this->track_items.back().num_repetitions =
                                        temp_num_repetitions;
                                }
                                else
                                {
                                    break;
                                }
                            }
                        }
                    }
                }

                return success;
            }

            void clear()
            {
                this->track_config = nullptr;

                if( this->path_particle_dump != nullptr )
                {
                    free( this->path_particle_dump );
                    this->path_particle_dump = nullptr;
                }

                if( this->path_lattice_dump != nullptr )
                {
                    free( this->path_lattice_dump );
                    this->path_lattice_dump = nullptr;
                }

                this->track_items.clear();
            }

            std::vector< TrackItem > track_items;
            ::toml_table_t* track_config;
            char* path_particle_dump;
            char* path_lattice_dump;
        };


        template< class TrackJobT >
        bool TrackJob_run_benchmark( TrackJobT& job,
            MainConfig const& main_config, TargetConfig const& target_conf,
            TrackConfig const& track_config )
        {
            bool success = false;

            namespace st = SIXTRL_CXX_NAMESPACE;

            auto const start_of_benchmark = std::chrono::system_clock::now();
            auto const begin_time = std::chrono::system_clock::to_time_t(
                start_of_benchmark );

            std::ostringstream a2str;

            a2str << std::put_time(
                std::localtime( &begin_time ), "%Y%m%d_%H%M%S" );

            std::string const begin_time_str = a2str.str();
            a2str.str( "" );

            if( main_config.output_path != nullptr )
            {
                a2str << main_config.output_path << "/";
            }
            else
            {
                a2str << "./";
            }

            std::string const output_dir = a2str.str();
            std::string name( "benchmark" );

            if( main_config.name_str != nullptr )
            {
                name = std::string{ main_config.name_str };
            }

            a2str << name << "_" << begin_time_str << ".log";
            std::string const path_log_file( a2str.str() );
            std::ofstream log_file( path_log_file.c_str() );

            a2str.str( output_dir.c_str() );
            a2str << name << "_" << begin_time_str << ".times";

            std::string const path_timing_file( a2str.str() );
            std::ofstream time_file( path_timing_file.c_str() );


            log_file << "generated by: run_benchmark_cxx\r\n"
                     << "name        : " << name << "\r\n"
                     << "started at  : " << begin_time_str << "\r\n"
                     << "arch_str    : " << target_conf.arch_str << "\r\n"
                     << "node_id_str : ";

            time_file << "# Generated by : run_benchmark_cxx\r\n"
                      << "# Benchmark name : " << name
                      << ", start at " << begin_time_str << "\r\n"
                      << "# arch_str : " << target_conf.arch_str
                      << ", node_id_str = ";

            if( target_conf.node_id_str != nullptr )
            {
                log_file  << target_conf.node_id_str << "\r\n";
                time_file << target_conf.node_id_str << "\r\n";
            }
            else
            {
                log_file  << "n/a\r\n";
                time_file << "n/a\r\n";
            }

            log_file  << "config_str  : ";

            if( target_conf.config_str != nullptr )
            {
                log_file  << target_conf.config_str << "\r\n";
                time_file << "#config_str : " << target_conf.config_str << "\r\n";
            }

            log_file << "git branch  : ";

            if( main_config.git_branch_str != nullptr )
            {
                log_file << main_config.git_branch_str << "\r\n";
            }
            else
            {
                log_file << "n/a\r\n";
            }

            log_file << "git hash    : ";

            if( main_config.git_hash_str != nullptr )
            {
                log_file << main_config.git_hash_str << "\r\n";
            }
            else
            {
                log_file << "n/a\r\n";
            }

            log_file  << "particles   : "
                      << track_config.path_particle_dump << "\r\n"
                      << "lattice     : "
                      << track_config.path_lattice_dump << "\r\n"
                      << "timings in  : " << path_timing_file << "\r\n\r\n";

            time_file << "#particles = "
                      << track_config.path_particle_dump << "\r\n"
                      << "#lattice = "
                      << track_config.path_lattice_dump << "\r\n"
                      << "#log file = " << path_log_file << "\r\n"
                      << "#\r\n" << "#"
                      << std::setw( 19 ) << "Num Part"
                      << std::setw( 20 ) << "Num Turns"
                      << std::setw( 20 ) << "Repetitions"
                      << std::setw( 20 ) << "Time/Part/Turn"
                      << std::setw( 20 ) << "Lost Particles"
                      << std::setw( 20 ) << "Min Time/Part/Turn"
                      << std::setw( 20 ) << "Lost Particles"
                      << std::setw( 20 ) << "Max Time/Part/Turn"
                      << std::setw( 20 ) << "Lost Particles"
                      << std::setw( 20 ) << "Total Time"
                      << "\r\n"
                      << "#"
                      << std::setw( 19 ) << "[#particles]"
                      << std::setw( 20 ) << "[#turns]"
                      << std::setw( 20 ) << "[#repetitions]"
                      << std::setw( 20 ) << "[sec]"
                      << std::setw( 20 ) << "[#particles]"
                      << std::setw( 20 ) << "[sec]"
                      << std::setw( 20 ) << "[#particles]"
                      << std::setw( 20 ) << "[sec]"
                      << std::setw( 20 ) << "[#particles]"
                      << std::setw( 20 ) << "[sec]"
                      <<"\r\n";

            SIXTRL_ASSERT( job.ptrParticlesBuffer() != nullptr );
            SIXTRL_ASSERT( job.ptrBeamElementsBuffer() != nullptr );

            st::Buffer init_pb( *job.ptrParticlesBuffer() );
            st::Buffer eb( *job.ptrBeamElementsBuffer() );
            st::Particles const* init_particles =
                init_pb.get< st::Particles >( 0 );

            SIXTRL_ASSERT( init_particles != nullptr );
            int64_t const init_num_particles = init_particles->getNumParticles();
            SIXTRL_ASSERT( init_num_particles > int64_t{ 0 } );

            std::random_device rdev;
            std::mt19937_64 prng( rdev() );

            std::normal_distribution< double > x_dist(
                init_particles->getXValue( 0 ), 1e-9 );

            std::normal_distribution< double > y_dist(
                init_particles->getYValue( 0 ), 1e-9 );

            std::normal_distribution< double > px_dist(
                init_particles->getPxValue( 0 ), 1e-12 );

            std::normal_distribution< double > py_dist(
                init_particles->getPyValue( 0 ), 1e-12 );

            std::normal_distribution< double > delta_dist(
                init_particles->getDeltaValue( 0 ), 1e-14 );

            std::normal_distribution< double > zeta_dist(
                init_particles->getZetaValue( 0 ), 1e-14 );

            auto track_it  = track_config.track_items.begin();
            auto track_end = track_config.track_items.end();

            for( ; track_it != track_end ; track_it++ )
            {
                using result_pair_t = std::pair< double, int64_t >;
                std::vector< result_pair_t > results(
                    track_it->num_repetitions, result_pair_t{ 0.0, 0 } );

                a2str.str( "" );

                auto start_setup_time = std::chrono::system_clock::now();
                auto time_obj = std::chrono::system_clock::to_time_t(
                    start_setup_time );

                a2str << std::put_time( std::localtime( &time_obj ),
                                        "%Y%m%d_%H%M%S" );

                log_file << "starting with track_until item at "
                         << a2str.str() << "\r\n"
                         << " -> num_particles   = "
                         << std::setw( 12 ) << track_it->num_particles << "\r\n"
                         << " -> num_turns       = "
                         << std::setw( 12 ) << track_it->num_turns << "\r\n"
                         << " -> num_repetitions = "
                         << std::setw( 12 ) << track_it->num_repetitions
                         << "\r\n" << "setup particle buffer ... ";

                int64_t const num_particles = track_it->num_particles;

                start_setup_time = std::chrono::system_clock::now();

                st::Buffer run_pb;
                st::Particles* particles =
                    run_pb.createNew< st::Particles >( num_particles );

                SIXTRL_ASSERT( particles != nullptr );
                for( int64_t ii = 0 ; ii < num_particles ; ++ii )
                {
                    using param_t =
                        std::normal_distribution< double >::param_type;

                    particles->copySingle( *init_particles,
                        ii % init_num_particles, ii );

                    particles->setXValue( ii, x_dist( prng, param_t(
                        particles->getXValue( ii ), x_dist.stddev() ) ) );

                    particles->setYValue( ii, y_dist( prng, param_t(
                        particles->getYValue( ii ), y_dist.stddev() ) ) );

                    particles->setPxValue( ii, px_dist( prng, param_t(
                        particles->getXValue( ii ), px_dist.stddev() ) ) );

                    particles->setPyValue( ii, py_dist( prng, param_t(
                        particles->getYValue( ii ), py_dist.stddev() ) ) );

                    particles->setDeltaValue( ii, px_dist( prng, param_t(
                        particles->getXValue( ii ), delta_dist.stddev() ) ) );

                    particles->setZetaValue( ii, py_dist( prng, param_t(
                        particles->getZetaValue( ii ), zeta_dist.stddev() ) ) );

                    particles->setParticleIdValue( ii, ii );
                    particles->setStateValue( ii, 1 );
                    particles->setAtElementIdValue( ii, 0 );
                    particles->setAtTurnValue( ii, 0 );
                }

                auto end_setup_time = std::chrono::system_clock::now();
                time_obj = std::chrono::system_clock::to_time_t( end_setup_time );

                a2str.str("");
                a2str << std::put_time( std::localtime( &time_obj ),
                                        "%Y%m%d_%H%M%S" );

                auto diff_setup = end_setup_time - start_setup_time;

                log_file << " finished at " << a2str.str() << "(took "
                    << std::chrono::duration< double >( diff_setup ).count()
                    << " sec)\r\n";

                log_file << "starting to track: ";
                auto start_repetitions_time = std::chrono::system_clock::now();
                start_setup_time = start_repetitions_time;
                for( int64_t ii = 0 ; ii < track_it->num_repetitions ; ++ii )
                {
                    if( ( ii > 0 ) && ( ( ii % 10 ) == 0 ) )
                    {
                        end_setup_time = std::chrono::system_clock::now();
                        diff_setup = end_setup_time - start_setup_time;
                        start_setup_time = end_setup_time;

                        log_file << " took " << std::chrono::duration< double >(
                            diff_setup ).count() << " sec\r\n";
                    }

                    st::Buffer pb( run_pb );
                    log_file << "resetting the track job with particles ... ";

                    start_setup_time = std::chrono::system_clock::now();
                    success = job.reset( pb, eb );
                    end_setup_time = std::chrono::system_clock::now();

                    diff_setup = end_setup_time - start_setup_time;

                    log_file << " finished at " << a2str.str() << "(took "
                        << std::chrono::duration< double >( diff_setup ).count()
                        << " sec)\r\n";

                    auto start_track_time = std::chrono::steady_clock::now();
                    job.trackUntil( track_it->num_turns );
                    auto end_track_time = std::chrono::steady_clock::now();

                    auto diff_track = end_track_time - start_track_time;
                    results[ ii ].first =
                        std::chrono::duration< double >( diff_track ).count();

                    for( int64_t jj = 0 ; jj < num_particles ; ++jj )
                    {
                        if( particles->getStateValue( jj ) == 0 )
                        {
                            ++results[ ii ].second;
                        }
                    }

                    log_file << "*";
                }

                auto end_repetitions_time = std::chrono::system_clock::now();
                diff_setup = end_repetitions_time - start_repetitions_time;
                time_obj = std::chrono::system_clock::to_time_t(
                    end_repetitions_time );

                a2str.str( "" );
                a2str << std::put_time( std::localtime( &time_obj ),
                                        "%Y%m%d_%H%M%S" );

                log_file << "\r\n"
                         << "finished all repetations at " << a2str.str()
                         << "\r\n";

                std::sort( results.begin(), results.end(),
                    []( result_pair_t const& lhs, result_pair_t const& rhs )
                    { return lhs.first < rhs.first; } );

                double const total_time = std::accumulate(
                    results.begin(), results.end(), double{ 0.0 },
                    []( double const sum, result_pair_t const& res_pair ) -> double
                    { return sum + res_pair.first; } );

                log_file << "took " << std::chrono::duration< double >(
                                diff_setup ).count()
                         << " sec \r\n"
                         << "time spent in tracking: " << total_time << " sec"
                         << "\r\n" << std::endl;

                double const norm_factor = static_cast< double >(
                    num_particles * track_it->num_turns );

                SIXTRL_ASSERT( norm_factor > 0.0 );

                for( auto& result_pair : results )
                {
                    result_pair.first /= norm_factor;
                }

                auto const median = ( results.size() + 1 ) >> 1;

                time_file << std::setw( 20 ) << num_particles
                          << std::setw( 20 ) << track_it->num_turns
                          << std::setw( 20 ) << track_it->num_repetitions
                          << std::setw( 20 ) << results[ median ].first
                          << std::setw( 20 ) << results[ median ].second
                          << std::setw( 20 ) << results[ 0 ].first
                          << std::setw( 20 ) << results[ 0 ].second
                          << std::setw( 20 ) << results.back().first
                          << std::setw( 20 ) << results.back().second
                          << std::setw( 20 ) << total_time
                          << std::endl;
            }

            return success;
        }
    }
}

int main( int argc, char* argv[] )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    if( argc < 2 )
    {
        std::cout << "Usage: " << argv[ 0 ]
                  << " path/to/config.toml \r\n" << std::endl;
        return 0;
    }

    st::benchmark::MainConfig   main_config( argv[ 1 ] );
    st::benchmark::TargetConfig target_config( main_config );
    st::benchmark::TrackConfig  track_config( main_config );

    if( ( track_config.path_particle_dump == nullptr ) ||
        ( track_config.path_lattice_dump  == nullptr ) )
    {
        std::cerr << "Error: Need both particle and lattice data"
                  << std::endl;
        return 0;
    }

    if( track_config.track_items.empty() )
    {
        std::cerr << "Error: No track items defined" << std::endl;
        return 0;
    }

    std::string output_path( "." );
    if( main_config.output_path != nullptr )
    {
        output_path = main_config.output_path;
    }

    st::Buffer particle_buffer( track_config.path_particle_dump );
    st::Buffer beam_elements_buffer( track_config.path_lattice_dump );

    if( target_config.arch_str != nullptr )
    {
        bool success = false;
        std::string config_str( "" );
        std::string node_id_str( "" );

        if( target_config.config_str != nullptr )
        {
            config_str = std::string{ target_config.config_str };
        }

        if( target_config.node_id_str != nullptr )
        {
            node_id_str = std::string{ target_config.node_id_str };
        }

        if( std::strcmp( target_config.arch_str, "cpu" ) == 0 )
        {
            st::TrackJobCpu job( particle_buffer, beam_elements_buffer,
                nullptr, std::size_t{ 0 }, config_str );

            success = st::benchmark::TrackJob_run_benchmark(
                job, main_config, target_config, track_config );
        }

        #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
            SIXTRACKLIB_ENABLE_MODULE_OPENCL  == 1
        else if( std::strcmp( target_config.arch_str, "opencl" ) == 0 )
        {
            st::TrackJobCl job( node_id_str, particle_buffer,
                beam_elements_buffer, nullptr, std::size_t{ 0 },
                target_config.config_str );

            success = st::benchmark::TrackJob_run_benchmark(
                job, main_config, target_config, track_config );
        }
        #endif /* OpenCL */

        #if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
                     SIXTRACKLIB_ENABLE_MODULE_CUDA  == 1
        else if( std::strcmp( target_config.arch_str, "cuda" ) == 0 )
        {
            st::CudaTrackJob job( node_id_str, particle_buffer,
                beam_elements_buffer, nullptr, std::size_t{ 0 }, config_str );

            success = st::benchmark::TrackJob_run_benchmark(
                job, main_config, target_config, track_config );
        }
        #endif /* Cuda */
    }

    return 0;
}

/* end: */
