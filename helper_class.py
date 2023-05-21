import shutil
from ase.parallel import *
from ase.io import read, write
from ase.atoms import np
import os, sys
import time
import shutil
from ase.optimize import BFGS, FIRE, QuasiNewton
from glob import glob
from ase.constraints import FixAtoms
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.optimize.gpmin import gpmin
from ase import Atoms
import time
from deepmd.calculator import DP
from ase.visualize import view
from ase.neb import NEB, NEBTools
import random 
import pickle
import time
from copy import deepcopy
import warnings
import matplotlib.pylab as pylab
from ase.neb import NEBTools
from ase.utils.forcecurve import fit_images
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (5, 4.5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.linewidth': 1.5}
pylab.rcParams.update(params)


class DP_NEB(object):


    def __init__(self, fmax, calc_dir, num_steps=5000):

      self.fmax = fmax
      self.num_steps = num_steps
      self.calc = calc_dir


    @staticmethod
    def make_guess_path(left_image, ts_image, right_image, left_count, 
                        right_count):
        
        left_image.pbc = True
        right_image.pbc = True
        ts_image.pbc = True
        c = ts_image.constraints

        images = [left_image]
        images += [left_image.copy() for i in range(left_count)]
        images += [ts_image]
        neb = NEB(images)
        neb.interpolate(mic=True)
        interp_left = images[1:-1]
        del images

        images = [ts_image]
        images += [ts_image.copy() for i in range(right_count)]
        images += [right_image]
        neb = NEB(images)
        neb.interpolate(mic=True)
        interp_right = images[1:-1]
        del images

        interp_path = interp_left + [ts_image] + interp_right
        for image in interp_path:
            image.set_constraint(c)
        
        return interp_path


    def check_ase_convergence(self, log_dir):

        with open(log_dir) as txt:
            for line in txt:
                ...
            last_line = line

        info = [last_line.split()[i] for i in [1, 4]]
        num_steps, fmax = int(info[0]), float(info[1])

        if fmax > self.fmax or num_steps > self.num_steps:
            return False, num_steps
        else:
            return True, num_steps


    def run_ase_opt(self, atoms, tag):

        ti = time.time()
        atoms.calc = DP(self.calc)
        optimizer = FIRE(atoms, logfile=tag+'_opt.log')
        optimizer.run(fmax=self.fmax, steps=self.num_steps) 
        E = atoms.get_potential_energy()
        convergence, num_steps = self.check_ase_convergence(tag+'_opt.log')

        if convergence is True:
          print('%s takes %s steps to converge. :)' % (tag, num_steps))
          tf = time.time()
          print(f'Calculation(s) finished in {(tf - ti)/60} minutes.')
          return atoms
        else:
          print('%s fails to converge within %s steps. :()' % self.num_steps)
          return None
          

    def run_ase_nebs(self, opt_is, guessed_ts, opt_fs, left_count, right_count):

        ti = time.time()
        if all([opt_is, opt_fs]) is not None:
            atoms_path = self.make_guess_path(opt_is, guessed_ts, opt_fs, 
                                              left_count, right_count)
                
            all_traj = [opt_is]
            for atoms in atoms_path:
                atoms.pbc = True
                atoms.calc = DP(self.calc)
                all_traj.append(atoms)
            all_traj.append(opt_fs)
                
            neb = NEB(all_traj, climb=True, parallel=True)
            neb.interpolate(mic=True) 
            optimizer = FIRE(neb, logfile='NEB.log')
            optimizer.run(fmax=self.fmax, steps=self.num_steps)
                
            if self.check_ase_convergence('NEB.log'):
                write('final_neb_from_deepmd.traj', all_traj)
                tf = time.time()
                print(f'Calculation(s) finished in {(tf - ti)/60} minutes.')
                return all_traj
            else:
                print('ASE NEB did not converge. :( ')
        else:
            print('Optimization did not converge. Abort NEB run. :( ')


    @staticmethod
    def plot_neb(atoms_path):

        fig, ax = plt.subplots()
        forcefit = fit_images(atoms_path)
        ax.plot([i for i in forcefit.path], [i for i in forcefit.energies], 'o', color='b')
        ax.plot([i for i in forcefit.fit_path], [i for i in forcefit.fit_energies], '-', color='r')
        for index, (x, y) in enumerate(forcefit.lines):
            if index == np.argmax(forcefit.energies):
                ax.plot([i for i in x], y, '-k')
        
        ax.set_xlabel(r'path [Å]')
        ax.set_ylabel('energy [eV]')

        plt.show()  





