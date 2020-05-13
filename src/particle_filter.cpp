/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::cout << "init:" << std::endl;
  num_particles = 25;  // TODO: Set the number of particles

  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);  

  for (int i=0; i<num_particles; ++i)
  {
    double sample_x, sample_y, sample_theta;
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    Particle p={i, sample_x, sample_y, sample_theta, 1.0};
    particles.push_back(p) ;
    weights.push_back(1.0);
  }
  // std::cout << "weights size:" << weights.size() << "particles" << particles.size() << std::endl;
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;

  for (int i=0; i<num_particles; ++i)
  {
    // std::cout << "yaw rate:" << yaw_rate << std::endl;
    // Calculate new prediction for each particle
    double pred_x, pred_y, pred_theta;
    if ((yaw_rate > -0.00001) && (yaw_rate < 0.00001)) // case 0: when yaw_rate very close to 0
    {
      pred_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      pred_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      pred_theta = particles[i].theta;
    }
    else // case 1: when yaw_rate is considerable
    {
      pred_x = particles[i].x 
                  + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      pred_y = particles[i].y
                  + velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      pred_theta = particles[i].theta + yaw_rate*delta_t;
    }
    // Add Gaussian noise
    double sample_x, sample_y, sample_theta;
    std::normal_distribution<double> dist_x(pred_x, std_pos[0]);
    std::normal_distribution<double> dist_y(pred_y, std_pos[1]);
    std::normal_distribution<double> dist_theta(pred_theta, std_pos[2]);
    
    // Update particle
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);  

    // std::cout << particles[i].x << " | " << particles[i].y << " | " << particles[i].theta << std::endl;

  }
   

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i=0; i<observations.size(); ++i)
  {
    // Loop through all predicted landmark and find the best match for the current observation
    double min_distance = INFINITY;
    double min_index = -1;

    for (int j=0; j<predicted.size(); ++j)
    {
      double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      if (distance < min_distance)
      {
        min_distance = distance;
        min_index = j;
      }      
    }

    if (min_index>=0)
    {
      // Assign the current observed measurement to this landmark
      observations[i].id = predicted[min_index].id;

      // Drop the chosen predicted landmark so it wont be picked again
      predicted.erase(predicted.begin()+min_index);
    }
    else
    {
      observations[i].id = -1;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i=0; i<num_particles; ++i)
  {
    // std::cout << "particle:" << i << std::endl;

    // Transform observations from VEHICLE's to MAP's coordinate
    vector<LandmarkObs> observed_landmarks;
    for (int j=0; j<observations.size(); ++j)
    {
      double x_map, y_map;
      x_map = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
      y_map = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
      LandmarkObs single_landmark = {-1, x_map, y_map};
      observed_landmarks.push_back(single_landmark);
    } 
    // std::cout << "observations size:" << observed_landmarks.size() << std::endl;
    // Based on sensor range, find predicted measurement
    vector<LandmarkObs> predicted_landmarks;
    for (int j=0; j<map_landmarks.landmark_list.size(); ++j)
    {
      // std::cout << particles[i].x << " | " << particles[i].y << " | " << map_landmarks.landmark_list[j].x_f << " | " << map_landmarks.landmark_list[j].y_f << std::endl;
      double distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      // std::cout << "dist:" << distance << std::endl;

      if (distance <= sensor_range)
      {
        LandmarkObs single_landmark = { map_landmarks.landmark_list[j].id_i, 
                                        map_landmarks.landmark_list[j].x_f, 
                                        map_landmarks.landmark_list[j].y_f
                                      };
        predicted_landmarks.push_back(single_landmark);
      }
    }
    // std::cout << "predicted size:" << predicted_landmarks.size() << std::endl;

    // Find associations
    dataAssociation(predicted_landmarks, observed_landmarks);    
    
    // Prepare arguments for SetAssociation function
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    // Calculate particle weight
    double sig_x, sig_y;
    sig_x = std_landmark[0];
    sig_y = std_landmark[1];

    double part_weight = 1;

    for (int t=0; t<observed_landmarks.size(); ++t)
    {
      double x_obs, y_obs, mu_x, mu_y;
      for (int k=0; k<predicted_landmarks.size(); ++k)
      {
        if (observed_landmarks[t].id == predicted_landmarks[k].id)
        {
          associations.push_back(observed_landmarks[t].id);
          sense_x.push_back(observed_landmarks[t].x);
          sense_y.push_back(observed_landmarks[t].y);

          // std::cout << "match:" << t << "->" << k << std::endl;
          x_obs = observed_landmarks[t].x;
          y_obs = observed_landmarks[t].y;
          mu_x = predicted_landmarks[k].x;
          mu_y = predicted_landmarks[k].y;

          double gauss_norm, exponent, single_weight;
          gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
          exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
                     + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
          single_weight = gauss_norm * exp(-exponent);
          part_weight *= single_weight;  
        }
      }
    }

    if (part_weight != 1)
    {
      particles[i].weight = part_weight;
      weights[i] = part_weight;
      SetAssociations(particles[i], associations, sense_x, sense_y);
    }

  }
  // std::cout << "weights size:" << weights.size() << "particles" << particles.size() << std::endl;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // std::cout << "resample" << std::endl;
  std::default_random_engine gen;
  std::discrete_distribution<int> d(weights.begin(), weights.end());
  std::vector<Particle> temp_particles;
  // std::cout << "weights size:" << weights.size() << "particles" << particles.size() << std::endl;

  for (int i=0; i<num_particles; ++i)
  {    
    int idx = d(gen);
    // std::cout << "idx:" << idx << std::endl;
    temp_particles.push_back(particles[d(gen)]);
  }
  // std::cout << "new particles size:" << temp_particles.size() << std::endl;
  particles = temp_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}