Physicians network, part of the Koblenz Network Collection
===========================================================================

This directory contains the TSV and related files of the moreno_innovation network: This directed network captures innovation spread among 246 physicians in for towns in Illinois, Peoria, Bloomington, Quincy and Galesburg. The data was collected in 1966. A node represents a physician and an edge between two physicians shows that the left physician told that the right physician is his friend or that he turns to the right physician if he needs advice or is interested in a discussion. There always only exists one edge between two nodes even if more than one of the listed conditions are true.


More information about the network is provided here: 
http://konect.cc/networks/moreno_innovation

Files: 
    meta.moreno_innovation -- Metadata about the network 
    out.moreno_innovation -- The adjacency matrix of the network in whitespace-separated values format, with one edge per line
      The meaning of the columns in out.moreno_innovation are: 
        First column: ID of from node 
        Second column: ID of to node
        Third column (if present): weight or multiplicity of edge
        Fourth column (if present):  timestamp of edges Unix time


Use the following References for citation:

@MISC{konect:2017:moreno_innovation,
    title = {Physicians network dataset -- {KONECT}},
    month = oct,
    year = {2017},
    url = {http://konect.cc/networks/moreno_innovation}
}

@article{konect:coleman1957,
	title = {The Diffusion of an Innovation Among Physicians},
	author = {Coleman, James and Katz, Elihu and Menzel, Herbert},
	journal = {Sociometry},
	pages = {253--270},
	year = {1957},
}

@article{konect:coleman1957,
	title = {The Diffusion of an Innovation Among Physicians},
	author = {Coleman, James and Katz, Elihu and Menzel, Herbert},
	journal = {Sociometry},
	pages = {253--270},
	year = {1957},
}


@inproceedings{konect,
	title = {{KONECT} -- {The} {Koblenz} {Network} {Collection}},
	author = {Jérôme Kunegis},
	year = {2013},
	booktitle = {Proc. Int. Conf. on World Wide Web Companion},
	pages = {1343--1350},
	url = {http://dl.acm.org/citation.cfm?id=2488173},
	url_presentation = {https://www.slideshare.net/kunegis/presentationwow},
	url_web = {http://konect.cc/},
	url_citations = {https://scholar.google.com/scholar?cites=7174338004474749050},
}

@inproceedings{konect,
	title = {{KONECT} -- {The} {Koblenz} {Network} {Collection}},
	author = {Jérôme Kunegis},
	year = {2013},
	booktitle = {Proc. Int. Conf. on World Wide Web Companion},
	pages = {1343--1350},
	url = {http://dl.acm.org/citation.cfm?id=2488173},
	url_presentation = {https://www.slideshare.net/kunegis/presentationwow},
	url_web = {http://konect.cc/},
	url_citations = {https://scholar.google.com/scholar?cites=7174338004474749050},
}


