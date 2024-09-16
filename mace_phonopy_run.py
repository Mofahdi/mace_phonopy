import argparse
import sys, os
from mace_phonopy_class import *
# packages needed:
# 1- torch
# 2- phonopy
# 3- ase
# 4- e3nn
# 5- mace-torch
#

def none_or_val(value):
	if value == 'None':
		return None
	return int(value)

def bool_vals(value):
	if value.lower() == 'true':
		return True
	if value.lower() == 'false':
		return False


parser = argparse.ArgumentParser(description='chgnet inputs')
parser.add_argument('--atoms_path', default='./POSCAR', type=str)
parser.add_argument('--supercell-dims','-scell-dims', default=["2", "2", "2"], type=list)
parser.add_argument('--disp', default=0.01, type=float)
parser.add_argument('--num_rand_disp', default=None, type=none_or_val)
parser.add_argument('--output_disp', default=True, type=bool_vals)

parser.add_argument('--pretrained_model', default=True, type=bool_vals)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--model_path', default='./', type=str)

parser.add_argument('--stability_criteria', default=-0.1, type=float)
parser.add_argument('--output_ph_band', default=True, type=bool_vals)

args = parser.parse_args(sys.argv[1:])

pmg_struc=Structure.from_file(args.atoms_path)
mace_phon=mace_phonopy(pmg_struc, path='.', supercell_dims=[int(args.supercell_dims[0]), int(args.supercell_dims[1]),int(args.supercell_dims[2])])
mace_phon.save_to_pickle()

#with open('chgnet_data.pkl', 'rb') as inp:
#	chgnet_obj= pickle.load(inp)
mace_phon.generate_bands_conf()
mace_phon.get_phonon_fc2(
			displacement=args.disp, 
			num_snapshots=args.num_rand_disp, 
			write_fc=True, 
			output_POSCARs=args.output_disp, 
			pretrained_model=args.pretrained_model, 
			default_dtype=args.dtype,
			device=args.device, 
			trained_path=args.model_path,
			)

mace_phon.get_phonon_dos_bs(
				line_density=40, 
				units='THz', 
				output_ph_band=args.output_ph_band, 
				stability_threshold=args.stability_criteria, 
				phonopy_bands_dos_figname='phonopy_bands_dos.png', 
				dpi=200,
				)
