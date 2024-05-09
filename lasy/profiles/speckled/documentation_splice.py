class _DocumentedMetaClass(type):
    """This is used as a metaclass that combines the __doc__ of the picmistandard base and of the implementation"""

    def __new__(cls, name, bases, attrs):
        # "if bases" skips this for the _ClassWithInit (which has no bases)
        # "if bases[0].__doc__ is not None" skips this for the picmistandard classes since their bases[0] (i.e. _ClassWithInit)
        # has no __doc__.
        if bases and bases[0].__doc__ is not None:
            implementation_doc = attrs.get("__doc__", "")
            # print('implementation doc',implementation_doc)
            base_doc = bases[0].__doc__
            param_delimiter = "Parameters\n    ----------\n"
            opt_param_delimiter = "    do_include_transverse_envelope"

            if implementation_doc:
                # The format of the added string is intentional.
                # The double return "\n\n" is needed to start a new section in the documentation.
                # Then the four spaces matches the standard level of indentation for doc strings
                # (assuming PEP8 formatting).
                # The final return "\n" assumes that the implementation doc string begins with a return,
                # i.e. a line with only three quotes, """.
                implementation_notes, implementation_params = implementation_doc.split(
                    param_delimiter
                )
                base_doc_notes, base_doc_params = base_doc.split(param_delimiter)
                base_doc_needed_params, base_doc_opt_params = base_doc_params.split(
                    opt_param_delimiter
                )
                attrs["__doc__"] = (
                    base_doc_notes
                    + implementation_notes
                    + param_delimiter
                    + base_doc_needed_params
                    + implementation_params[:-5]
                    + "\n\n"
                    + opt_param_delimiter
                    + base_doc_opt_params
                )
            else:
                attrs["__doc__"] = base_doc
        return super(_DocumentedMetaClass, cls).__new__(cls, name, bases, attrs)
