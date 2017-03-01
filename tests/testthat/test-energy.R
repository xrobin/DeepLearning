context("energy")

test.dat <- mnist$test$x[1:10,]

test_that("energy.dbn works as expected", {
	en <- energy(trained.mnist, test.dat)
	expected_energy <- c(-1496.25214569678, -1597.91150630306, -759.82194851186, -3078.17123282436, 
						-1100.33148621083, -1523.65373744315, -1382.71141763502, -1140.53360149193, 
						-2387.52745725336, -2651.44586886771)
	expect_equal(en, expected_energy) # somehow expected_error is a bit imprecise

	# Works with 1 row
	expect_identical(energy(trained.mnist, test.dat[1,, drop = FALSE]), en[1])
})


test_that("energy.rbm works as expected", {
	rbm <- pretrained.mnist[[1]]
	en <- energy(rbm, test.dat)
	expected_energy <- c(-1103.16590653693, -1241.73981978846, -391.214276304729, -2357.29932328728, 
						 -758.216033850546, -978.69288214803, -1051.10057508992, -856.718569750233, 
						 -1966.14552962485, -2029.19683106069)
	expect_equal(en, expected_energy) # somehow expected_error is a bit imprecise

	# Works with 1 row
	expect_identical(energy(rbm, test.dat[1,, drop = FALSE]), en[1])
})


test_that("energy.dbn errors if passed invalid data", {
	# Don't accept a vector
	expect_error(energy(trained.mnist, test.dat[1,, drop = TRUE]))
	expect_error(energy(pretrained.mnist[[1]], test.dat[1,, drop = TRUE]))
	
	# Don't accept wrong dimensions
	expect_error(energy(trained.mnist, test.dat[, 1:20, drop = FALSE]), regexp = "column")
	expect_error(energy(trained.mnist[[1]], test.dat[, 1:20, drop = FALSE]), regexp = "column")
})
