import os
import numpy as np
import pickle

from phonopy import Phonopy
from phonopy.file_IO import (
				write_FORCE_CONSTANTS,
				)

#from chgnet.model.model import CHGNet
from mace.calculators import mace_mp
from mace.calculators import MACECalculator

from pymatgen.core import Structure
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from jarvis.core.kpoints import Kpoints3D

from pymatgen.io.phonopy import get_pmg_structure, get_phonopy_structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atoms as AseAtoms
from phonopy.structure.atoms import Atoms as PhonopyAtoms


''' ase atoms to phonopy atoms '''
def ase_to_phonopy_atoms(ase_atoms, pbc=True):
	return PhonopyAtoms(
			symbols=ase_atoms.symbols,
			positions=ase_atoms.get_positions(),
			pbc=pbc,
			cell=ase_atoms.get_cell(),
			)

''' phonopy atoms to ase atoms '''
def phonopy_to_ase_atoms(phonopy_atoms, pbc=True):
	return AseAtoms(
		symbols=phonopy_atoms.symbols,
		positions=phonopy_atoms.positions,
		pbc=pbc,
		cell=phonopy_atoms.cell,
		)

''' mace phonopy object '''
class mace_phonopy:
	def __init__(
			self, 
			structure: Structure, 
			path='.', 
			supercell_dims=[2,2,2],
			):
		self.structure=structure
		self.phonopy_structure=get_phonopy_structure(self.structure)
		self.jarvis_atoms=JarvisAtomsAdaptor.get_atoms(self.structure)
	
		self.path=path
	
		self.supercell_dims=supercell_dims
		# create supercell attribute in object through the supercell function
		self.supercell=self.create_supercell()	

	
	''' function to create supercell '''
	def create_supercell(
			self
			):
		new_structure=self.structure.copy()
		new_structure.make_supercell(self.supercell_dims)
		supercell_name=os.path.join(self.path, 'SPOSCAR_'+str(self.supercell_dims[0])+str(self.supercell_dims[1])+str(self.supercell_dims[2]))
		new_structure.to(filename=supercell_name)
		return new_structure

	
	''' generate kpoints '''
	def get_jarvis_kpoints(
				self, 
				line_density=20,
				):
		kpoints = Kpoints3D().kpath(self.jarvis_atoms, line_density=line_density)
		return kpoints


	''' function to save object using pickle '''
	def save_to_pickle(
				self, 
				filename='mace_phonopy_attrs.pkl',
				):
		filename=os.path.join(self.path, filename)
		with open(filename, 'wb') as outp:
			pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


	''' function to generate displacements and 2nd order IFC. The IFC is saved as object attribute '''
	def get_phonon_fc2(
				self, 
				displacement=0.01, 
				num_snapshots=None,
				write_fc=True,
				output_POSCARs=False, 
				pretrained_model=True,
				default_dtype="float64",
				device='cpu', 
				trained_path='./MACE.model',
				):
		if pretrained_model:
			calc = mace_mp(model="medium-mpa-0", dispersion=False, default_dtype=default_dtype, device=device)
		else:
			calc = MACECalculator(model_path=trained_path, model_paths=None, device=device)

		phonon = Phonopy(self.phonopy_structure, [[self.supercell_dims[0], 0, 0], [0, self.supercell_dims[1], 0], [0, 0, self.supercell_dims[2]]])
		phonon.generate_displacements(distance=displacement, number_of_snapshots=num_snapshots)
		supercells = phonon.get_supercells_with_displacements()
		set_of_forces = []
		disp = 0
		for i_scell, scell in enumerate(supercells):
			#scell_pmg=get_pmg_structure(scell)
			scell_ase=phonopy_to_ase_atoms(scell, pbc=True)
			if output_POSCARs:
				scell_ase.write('POSCAR-'+"{0:0=3d}".format(i_scell+1), format='vasp', direct='True'); print("{0:0=3d}".format(i_scell+1))#f"i_scell+1:03d"
			#scell_predictions=chgnet.predict_structure(scell_pmg)
			#forces = np.array(scell_predictions['f'])
			scell_ase.calc=calc
			forces=scell_ase.get_forces()
			disp = disp + 1
			drift_force = forces.sum(axis=0)
			for force in forces:
				force -= drift_force / forces.shape[0]
			set_of_forces.append(forces)
	
		phonon.produce_force_constants(forces=set_of_forces)

		if write_fc:
			write_FORCE_CONSTANTS(
						phonon.get_force_constants(), filename="FORCE_CONSTANTS"
						)

		# save the phonon attribute to the object
		self.phonon=phonon

	
	''' method to output the phonon dispersion and DOS '''
	def get_phonon_dos_bs(
				self, 
				line_density=30, 
				units='THz', 
				output_ph_band: bool = True,
				stability_threshold=-0.1, 
				phonopy_bands_dos_figname='phonopy_bands_dos.png', 
				dpi=200, 
				):

		# freq_conversion_factor=1 # THz units
		# freq_conversion_factor=333.566830  # ThztoCm-1
		if units=='cm-1':
			freq_conversion_factor=333.566830
		else:
			freq_conversion_factor=1

		kpoints=self.get_jarvis_kpoints(line_density=line_density)
		lbls = kpoints.labels
		lbls_ticks = []
		freqs = []
		tmp_kp = []
		lbls_x = []
		count = 0
		stability=True
		for ii, k in enumerate(kpoints.kpts):
			k_str = ",".join(map(str, k))
			if ii == 0:
				tmp = []
				for i, freq in enumerate(self.phonon.get_frequencies(k)):
					tmp.append(freq)
					#print(freq)
					#for fs in freq:
					if freq < stability_threshold*freq_conversion_factor:
						stability=False
					
				freqs.append(tmp)
				tmp_kp.append(k_str)
				lbl = "$" + str(lbls[ii]) + "$"
				lbls_ticks.append(lbl)
				lbls_x.append(count)
				count += 1
				# lbls_x.append(ii)

			elif k_str != tmp_kp[-1]:
				tmp_kp.append(k_str)
				tmp = []
				for i, freq in enumerate(self.phonon.get_frequencies(k)):
					tmp.append(freq)
					
					#for fs in freq:
					if freq < stability_threshold*freq_conversion_factor:
						stability=False

				freqs.append(tmp)
				lbl = lbls[ii]
				if lbl != "":
					lbl = "$" + str(lbl) + "$"
					lbls_ticks.append(lbl)
					# lbls_x.append(ii)
					lbls_x.append(count)
				count += 1
				# lbls_x = np.arange(len(lbls_ticks))


		with open(os.path.join(self.path, 'stability'), 'w') as stable_file:
			if stability==True: 
				stable_file.write('stable'); #print('stable')
			elif stability==False: 
				stable_file.write('unstable'); #print('unstable')
		
		if output_ph_band:
			freqs = np.array(freqs)
			freqs = freqs * freq_conversion_factor
			# print('freqs',freqs,freqs.shape)
			the_grid = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.0)
			plt.rcParams.update({"font.size": 18})

			plt.figure(figsize=(10, 5))
			plt.subplot(the_grid[0])
			for i in range(freqs.shape[1]):
				plt.plot(freqs[:, i], lw=2, c="b")
			for i in lbls_x:
				plt.axvline(x=i, c="black")
			plt.xticks(lbls_x, lbls_ticks)
			# print('lbls_x',lbls_x,len(lbls_x))
			# print('lbls_ticks',lbls_ticks,len(lbls_ticks))
	
			if units=='cm-1':
				plt.ylabel("Frequency (cm$^{-1}$)")
			else: 
				plt.ylabel("Frequency (THz)")

			plt.xlim([0, max(lbls_x)])

			self.phonon.run_mesh([40, 40, 40], is_gamma_center=True, is_mesh_symmetry=False)
			self.phonon.run_total_dos()
			tdos = self.phonon._total_dos

			# print('tods',tdos._frequencies.shape)
			freqs, ds = tdos.get_dos()
			freqs = np.array(freqs)
			freqs = freqs * freq_conversion_factor
			min_freq = -0.05 * freq_conversion_factor
			max_freq = max(freqs)
			plt.ylim([min_freq, max_freq])

			plt.subplot(the_grid[1])
			plt.fill_between(
					ds, freqs, color=(0.2, 0.4, 0.6, 0.6), edgecolor="k", lw=1, y2=0
					)
			plt.xlabel("DOS")
			# plt.plot(ds,freqs)
			plt.yticks([])
			plt.xticks([])
			plt.ylim([min_freq, max_freq])
			plt.xlim([0, max(ds)])
			plt.tight_layout()
			plt.savefig(os.path.join(self.path, phonopy_bands_dos_figname), dpi=dpi)
			plt.close()

	''' generate band.conf files to run produce phonon dispersion ''' 
	def generate_bands_conf(
		self,
		filename="orig_band.conf",
		line_density=30,
		BAND_POINTS=100,
		):
		kpoints = Kpoints3D().kpath(self.jarvis_atoms, line_density=line_density)
		all_kp = kpoints._kpoints
		labels = kpoints._labels
		all_labels = ""
		all_lines = ""
		for lb in labels:
			if lb == "":
				lb = None
			all_labels = all_labels + str(lb) + str(" ")
		for k in all_kp:
			all_lines = (
				all_lines
				+ str(k[0])
				+ str(" ")
				+ str(k[1])
				+ str(" ")
				+ str(k[2])
				+ str(" ")
				)
		file = open(os.path.join(self.path, filename), "w")
		file.write('PRIMITIVE_AXES = AUTO\n')
		line = str("ATOM_NAME = ") + str(" ".join(list(set(self.jarvis_atoms.elements)))) + "\n"
		file.write(line)
		line = str("DIM = ") + " ".join(map(str, self.supercell_dims)) + "\n"
		file.write(line)
		line = str("FORCE_CONSTANTS = READ") + "\n"
		file.write(line)
		line = str("BAND= ") + str(all_lines) + "\n"
		file.write(line)
		#line = str("BAND_LABELS= ") + str(all_labels) + "\n"
		#file.write(line)
		file.close()


		ase_atoms=AseAtomsAdaptor.get_atoms(self.structure)
		bandpath_kpts = ase_atoms.cell.bandpath()._kpts
		kpts_str=str()
		for i in bandpath_kpts:
			kpts_str+=str(i[0])+' '+str(i[1])+' '+ str(i[2])+'  '

		file = open(os.path.join(self.path, 'band.conf'), "w")
		file.write('PRIMITIVE_AXES = AUTO\n')
		line = str("ATOM_NAME = ") + str(" ".join(list(set(self.jarvis_atoms.elements)))) + "\n"
		file.write(line)
		line = str("DIM = ") + " ".join(map(str, self.supercell_dims)) + "\n"
		file.write(line)
		line = str("FORCE_CONSTANTS = READ") + "\n"
		file.write(line)
		line = str("BAND= ") + kpts_str + "\n"
		file.write(line)
		file.write('BAND_POINTS = '+str(BAND_POINTS)+'\n')
		#line = str("BAND_LABELS= ") + str(all_labels) + "\n"
		#file.write(line)
		file.close()



if __name__=="__main__":
	pmg_struc=Structure.from_file('POSCAR')
	chg_phon=chgnet_phonopy(pmg_struc)
	chg_phon.save_to_pickle()

	#with open('chgnet_data.pkl', 'rb') as inp:
	#	chgnet_obj= pickle.load(inp)

	chg_phon.get_phonon_fc2()
	chg_phon.get_phonon_dos_bs()
	
