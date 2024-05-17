import numpy as np


class PicoOutputInfo:

    def __init__(self, filename):
        """
        The object is initialized with a full path
        to a pico_output_info.txt file, e.g.,

        filename = '/llust21/cosmo-plasm/zoom-simulations-arepo2/halo_0194/tng/zoom12/output/pico_output_info.txt'
        """

        dat = np.loadtxt(filename)

        self.num_vec = dat[:, 0].astype(int)
        self.full_vec = dat[:, 1].astype(bool)
        self.a_vec = dat[:, 2]
        self.z_vec = dat[:, 3]
        self.age_vec = dat[:, 4]

        self.num_vec_full = self.num_vec[self.full_vec]
        self.a_vec_full = self.a_vec[self.full_vec]
        self.z_vec_full = self.z_vec[self.full_vec]
        self.age_vec_full = self.age_vec[self.full_vec]

    def is_full(self, snapnum):
        return self.full_vec[snapnum]

    def closest_redshift(self, z, onlyfull=True):
        """
        Input: z, the redshift of interest.
        Returns the snapnumber closest in redshift-space,
                and its redshift. If onlyfull is True,
                then we only consider snap numbers for which
                we have full snapshots.
        """
        if not onlyfull:
            index = np.argmin(np.abs(self.z_vec - z))
            return index, self.z_vec[index]
        else:
            index = np.argmin(np.abs(self.z_vec_full - z))
            snapnum = self.num_vec_full[index]
            return snapnum, self.z_vec[snapnum]

    def closest_scalefactor(self, a, onlyfull=True):
        """
        Identical to 'closest_redshift' but for scale
        factors.
        """
        if not onlyfull:
            index = np.argmin(np.abs(self.a_vec - a))
            return index, self.a_vec[index]
        else:
            index = np.argmin(np.abs(self.a_vec_full - a))
            snapnum = self.num_vec_full[index]
            return snapnum, self.a_vec[snapnum]

    def closest_age(self, age, onlyfull=True):
        """
        Identical to 'closest_redshift' but for the
        age as a float in Gyrs.
        """
        if not onlyfull:
            index = np.argmin(np.abs(self.age_vec - age))
            return index, self.age_vec[index]
        else:
            index = np.argmin(np.abs(self.age_vec_full - age))
            snapnum = self.num_vec_full[index]
            return snapnum, self.age_vec[snapnum]
