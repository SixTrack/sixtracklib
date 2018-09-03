#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/common/particles.hpp"
#include "sixtracklib/common/buffer.hpp"

/* ========================================================================= */
/* ====  Test random initialization of particles                             */

TEST( CXX_ParticlesTests, RandomInitParticlesCopyAndCompare )
{
    namespace st = sixtrack;
    using size_t = st::Buffer::size_type;

    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );

    st::Buffer pb( size_t{ 1u << 20u } );

    size_t const NUM_PARTICLES = size_t{ 1000u };

    st::Particles* p = pb.createNew< st::Particles >( NUM_PARTICLES );

    ASSERT_TRUE( p != nullptr );
    ASSERT_TRUE( pb.getNumObjects() == size_t{ 1u } );

    ASSERT_TRUE( p->getNumParticles() == NUM_PARTICLES );
    ASSERT_TRUE( p->getQ0()           != nullptr );
    ASSERT_TRUE( p->getMass0()        != nullptr );
    ASSERT_TRUE( p->getBeta0()        != nullptr );
    ASSERT_TRUE( p->getGamma0()       != nullptr );
    ASSERT_TRUE( p->getP0c()          != nullptr );
    ASSERT_TRUE( p->getS()            != nullptr );
    ASSERT_TRUE( p->getX()            != nullptr );
    ASSERT_TRUE( p->getY()            != nullptr );
    ASSERT_TRUE( p->getPx()           != nullptr );
    ASSERT_TRUE( p->getPy()           != nullptr );
    ASSERT_TRUE( p->getZeta()         != nullptr );
    ASSERT_TRUE( p->getPSigma()       != nullptr );
    ASSERT_TRUE( p->getDelta()        != nullptr );
    ASSERT_TRUE( p->getRpp()          != nullptr );
    ASSERT_TRUE( p->getRvv()          != nullptr );
    ASSERT_TRUE( p->getChi()          != nullptr );
    ASSERT_TRUE( p->getParticleId()   != nullptr );
    ASSERT_TRUE( p->getAtElementId()  != nullptr );
    ASSERT_TRUE( p->getAtTurn()       != nullptr );
    ASSERT_TRUE( p->getState()        != nullptr );

    /* --------------------------------------------------------------------- */

    st::Particles* p_copy = pb.createNew< st::Particles >( NUM_PARTICLES );

    ASSERT_TRUE( p_copy != nullptr );
    ASSERT_TRUE( pb.getNumObjects() == size_t{ 2 } );
    ASSERT_TRUE( p_copy->getNumParticles() == NUM_PARTICLES );

    ASSERT_TRUE( p_copy->getQ0()          != nullptr );
    ASSERT_TRUE( p_copy->getMass0()       != nullptr );
    ASSERT_TRUE( p_copy->getBeta0()       != nullptr );
    ASSERT_TRUE( p_copy->getGamma0()      != nullptr );
    ASSERT_TRUE( p_copy->getP0c()         != nullptr );
    ASSERT_TRUE( p_copy->getS()           != nullptr );
    ASSERT_TRUE( p_copy->getX()           != nullptr );
    ASSERT_TRUE( p_copy->getY()           != nullptr );
    ASSERT_TRUE( p_copy->getPx()          != nullptr );
    ASSERT_TRUE( p_copy->getPy()          != nullptr );
    ASSERT_TRUE( p_copy->getZeta()        != nullptr );
    ASSERT_TRUE( p_copy->getPSigma()      != nullptr );
    ASSERT_TRUE( p_copy->getDelta()       != nullptr );
    ASSERT_TRUE( p_copy->getRpp()         != nullptr );
    ASSERT_TRUE( p_copy->getRvv()         != nullptr );
    ASSERT_TRUE( p_copy->getChi()         != nullptr );
    ASSERT_TRUE( p_copy->getParticleId()  != nullptr );
    ASSERT_TRUE( p_copy->getAtElementId() != nullptr );
    ASSERT_TRUE( p_copy->getAtTurn()      != nullptr );
    ASSERT_TRUE( p_copy->getState()       != nullptr );

    ASSERT_TRUE( p_copy->getQ0()          != p->getQ0() );
    ASSERT_TRUE( p_copy->getMass0()       != p->getMass0()  );
    ASSERT_TRUE( p_copy->getBeta0()       != p->getBeta0()  );
    ASSERT_TRUE( p_copy->getGamma0()      != p->getGamma0() );
    ASSERT_TRUE( p_copy->getP0c()         != p->getP0c()    );
    ASSERT_TRUE( p_copy->getS()           != p->getS()      );
    ASSERT_TRUE( p_copy->getX()           != p->getX()      );
    ASSERT_TRUE( p_copy->getY()           != p->getY()      );
    ASSERT_TRUE( p_copy->getPx()          != p->getPx()     );
    ASSERT_TRUE( p_copy->getPy()          != p->getPy()     );
    ASSERT_TRUE( p_copy->getZeta()        != p->getZeta()   );
    ASSERT_TRUE( p_copy->getPSigma()      != p->getPSigma() );
    ASSERT_TRUE( p_copy->getDelta()       != p->getDelta()  );
    ASSERT_TRUE( p_copy->getRpp()         != p->getRpp()    );
    ASSERT_TRUE( p_copy->getRvv()         != p->getRvv()    );
    ASSERT_TRUE( p_copy->getChi()         != p->getChi()    );
    ASSERT_TRUE( p_copy->getParticleId()  != p->getParticleId()  );
    ASSERT_TRUE( p_copy->getAtElementId() != p->getAtElementId() );
    ASSERT_TRUE( p_copy->getAtTurn()      != p->getAtTurn() );
    ASSERT_TRUE( p_copy->getState()       != p->getState()  );

    /* --------------------------------------------------------------------- */

    st_Particles_random_init( p );
    st_Particles_copy( p_copy, p );

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( st_Particles_have_same_structure( p_copy, p ) );
    ASSERT_TRUE( !st_Particles_map_to_same_memory( p_copy, p ) );
    ASSERT_TRUE( 0 == st_Particles_compare_values( p_copy, p ) );
}


/* end: tests/sixtracklib/common/test_particles_cxx.cpp */
