/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

std::default_random_engine gGen; //!< global random engine

// function for measuring 2D distane between 2 points
static double distance(double x1, double y1, double x2, double y2) {
	return ::sqrt((x1-x2) * (x1-x2) + (y1-y2) * (y1-y2));
}

// function for multivariate Gaussian distribution 2D
static double multivariate_gaussian(double mx, double my, double sigx, double sigy) {
	return (1. / (2. * M_PI * sigx * sigy)) * 
		::exp(-0.5 * (mx / sigx) * (mx / sigx) - 0.5 * (my / sigy) * (my / sigy));
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50;

	std::normal_distribution<double> dist_x(0, std[0]);
	std::normal_distribution<double> dist_y(0, std[1]);
	std::normal_distribution<double> dist_theta(0, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle p;
		p.id = i;
		p.x = x + dist_x(gGen);
		p.y = y + dist_y(gGen);
		p.theta = theta + dist_theta(gGen);
		p.weight = 1.;
		particles.emplace_back(p);
	}

	is_initialized = true;
	return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);

	if (::fabs(yaw_rate) < 0.00001) {
		for (auto & p : particles) {
			p.x = p.x + velocity * delta_t * ::cos(p.theta) + dist_x(gGen);
			p.y = p.y + velocity * delta_t * ::sin(p.theta) + dist_y(gGen);
			p.theta = p.theta + dist_theta(gGen);
		}
	}
	else {
		for (auto & p : particles) {
			p.x = p.x + (velocity/yaw_rate) * 
					(::sin(p.theta + yaw_rate * delta_t) - ::sin(p.theta)) + dist_x(gGen);
			p.y = p.y + (velocity/yaw_rate) * 
					(::cos(p.theta) - ::cos(p.theta + yaw_rate * delta_t)) + dist_y(gGen);
			p.theta = p.theta + yaw_rate * delta_t + dist_theta(gGen);
		}
	}
	return;
}

void ParticleFilter::dataAssociation(double sensor_range, std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (auto & obs : observations) {
		double min_dist = 2 * sensor_range;
		int id_min = -1;
		
		for (const auto & p : predicted) {
			double dist = distance(obs.x, obs.y, p.x, p.y);
			if (dist < min_dist) {
				min_dist = dist;
				id_min = p.id;
			}
		}
		obs.id = id_min;
	}
	return;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	// For each particle:
	for (auto & p : particles) {
		// 1. Find all landmarks that are within of sensor reach
		std::vector<LandmarkObs> predicted;
		for (const auto & ml : map_landmarks.landmark_list) {
			if (distance(p.x, p.y, ml.x_f, ml.y_f) <= sensor_range) {
				predicted.emplace_back(LandmarkObs{ml.id_i, ml.x_f, ml.y_f});
			}
		}

		// 2. Transform measurement observations from particle coordinate system to global coordinate system
		std::vector<LandmarkObs> global_obs;
		for (const auto & obs : observations) {
			double g_obs_x = ::cos(p.theta) * obs.x - ::sin(p.theta) * obs.y + p.x;
			double g_obs_y = ::sin(p.theta) * obs.x + ::cos(p.theta) * obs.y + p.y;
			global_obs.emplace_back(LandmarkObs{obs.id, g_obs_x, g_obs_y});
		}

		// 3. Associate landmark with each transformed measurement observation
		dataAssociation(sensor_range, predicted, global_obs);

		// 4. Compute new weight given all observations
		p.weight = 1.;
		
		for (const auto & go : global_obs) {
			//find associated prediction
			double pred_x = 0.;
			double pred_y = 0.;
			bool found = false;
			for (const auto & p : predicted) {
				if (p.id == go.id) {
					pred_x = p.x;
					pred_y = p.y;
					found = true;
					break;//-->
				}
			}
			
			if (found) {
				p.weight *= multivariate_gaussian(pred_x - go.x, pred_y - go.y, 
								std_landmark[0], std_landmark[1]);
			}
		}
	}
	return;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Implementation of resampling whell from particle filters lesson
	std::vector<Particle> new_particles;

	// Random starting index
	std::uniform_int_distribution<int> ind_dist(0, particles.size() - 1);
	auto ind = ind_dist(gGen);

	std::vector<double> weights;
	for (const auto & p : particles) weights.emplace_back(p.weight);

	// Max weight and beta distribution
	double max_weight = *std::max_element(weights.begin(), weights.end());
	std::uniform_real_distribution<double>  weight_dist(0., 2. * max_weight);

	//resampling wheel
	double beta = 0.;
	for (const auto & p : particles) {
		beta += weight_dist(gGen);
		while (beta > weights[ind]) {
			beta -= weights[ind];
			ind = (ind + 1) % particles.size();
		}
		new_particles.emplace_back(particles[ind]);
	}

	particles = new_particles;
	return;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
