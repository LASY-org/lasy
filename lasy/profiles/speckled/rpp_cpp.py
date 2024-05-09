import numpy as np
from .speckle_profile import SpeckleProfile

class _DocumentedMetaClass(type):
    """This is used as a metaclass that combines the __doc__ of the picmistandard base and of the implementation"""
    def __new__(cls, name, bases, attrs):
        # "if bases" skips this for the _ClassWithInit (which has no bases)
        # "if bases[0].__doc__ is not None" skips this for the picmistandard classes since their bases[0] (i.e. _ClassWithInit)
        # has no __doc__.
        if bases and bases[0].__doc__ is not None:
            implementation_doc = attrs.get('__doc__', '')
            # print('implementation doc',implementation_doc)
            base_doc = bases[0].__doc__
            param_delimiter = 'Parameters\n    ----------\n'
            opt_param_delimiter = 'do_include_transverse_envelope'

            if implementation_doc:
                # The format of the added string is intentional.
                # The double return "\n\n" is needed to start a new section in the documentation.
                # Then the four spaces matches the standard level of indentation for doc strings
                # (assuming PEP8 formatting).
                # The final return "\n" assumes that the implementation doc string begins with a return,
                # i.e. a line with only three quotes, """.
                implementation_notes, implementation_params = implementation_doc.split(param_delimiter)
                # print('implementation intro',implementation_notes)
                # print('implementation params', implementation_params)
                base_doc_notes, base_doc_params = base_doc.split(param_delimiter)
                base_doc_needed_params, base_doc_opt_params = base_doc_params.split(opt_param_delimiter)
                # print('base doc notes', base_doc_notes)
                # print('base doc needed params', base_doc_needed_params)
                # print('base doc opt params',base_doc_opt_params)
                print('implementation params',repr(implementation_params))
                attrs['__doc__'] = base_doc_notes + implementation_notes + param_delimiter + base_doc_needed_params# + implementation_params
                # attrs['__doc__'] = base_doc_notes + implementation_notes + param_delimiter + base_doc_needed_params + '\n' + implementation_params + '\n    ' + opt_param_delimiter + base_doc_opt_params
                # attrs['__doc__'] = bases[0].__doc__ + """\n\n    Implementation specific documentation\n""" + implementation_doc
                print('New doc:\n--------')
                print(repr(attrs['__doc__']))
            else:
                attrs['__doc__'] = base_doc
        return super(_DocumentedMetaClass, cls).__new__(cls, name, bases, attrs)
    
class PhasePlateProfile(SpeckleProfile, metaclass=_DocumentedMetaClass):
    r"""Generate a speckled laser profile with a random phase plate.

    This has no temporal smoothing.
    The amplitude of the beamlets is always :math:`A_{ml}(t)=1` and
    the relative phases of the beamlets, resulting from the randomly sized phase plate sections,
    are assigned randomly.
    If the user specifies Random Phase Plate (RPP: `rpp`), the beamlet phases are drawn with equal probabilities from the set :math:`{0,2\pi}`.
    If the user specifies Continuous Phase Plate (CPP: `cpp`), the beamlet phases are drawn from a uniform distribution on the interval :math:`[0,2\pi]`.

    Parameters
    ----------
    rpp_cpp: string, can be 'rpp' or 'cpp'
        Whether to assign beamlet phases according to RPP or CPP scheme
    """
    
    def __init__(
        self,
        wavelength,
        pol,
        laser_energy,
        focal_length,
        beam_aperture,
        n_beamlets,
        rpp_cpp,
        do_include_transverse_envelope=True,
        long_profile=None
    ):
        super().__init__(
            wavelength,
            pol,
            laser_energy,
            focal_length,
            beam_aperture,
            n_beamlets,
            do_include_transverse_envelope,
            long_profile,
        )
        self.rpp_cpp = rpp_cpp

    def beamlets_complex_amplitude(
        self,
        t_now,
    ):
        """Calculate complex amplitude of the beamlets in the near-field, before propagating to the focal plane.

        Parameters
        ----------
        t_now: float, time at which to evaluate complex amplitude

        Returns
        -------
        array of complex numbers giving beamlet amplitude and phases in the near-field
        """
        if self.rpp_cpp.upper() == "RPP":
            phase_plate = np.random.choice([0, np.pi], self.n_beamlets)
        else:
            phase_plate = np.random.uniform(
                -np.pi, np.pi, size=self.n_beamlets[0] * self.n_beamlets[1]
            ).reshape(self.n_beamlets)
        exp_phase_plate = np.exp(1j * phase_plate)
        return exp_phase_plate
