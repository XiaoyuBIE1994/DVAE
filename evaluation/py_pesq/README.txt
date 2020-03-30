--------------------------------------------------------------
*** README file for ITU-T P.862 Annex A (2005) Software
*** Reference implementation of P.862, P.862.1 and P.862.2 and 
*** conformance data for P.862 and P.862.2
*** Version 2.0 (October 2005)
--------------------------------------------------------------

RECOMMENDATION TITLES
---------------------
 ITU-T P.862 (02/2001) - Perceptual evaluation of speech quality
     (PESQ), an objective method for end-to-end speech quality
     assessment of narrowband telephone networks and speech codecs.

 ITU T P.862 Annex A (2005), Reference implementations and conformance 
     testing for Recommendations P.862, P.862.1 and P.862.2.

 ITU T P.862.1 (2003), Mapping function for transforming P.862 raw 
     result scores to MOS-LQO.

 ITU T P.862.2 (2005) Wideband extension to P.862 for the assessment 
     of wideband telephone networks and speech codecs.


              ****************************************
              PESQ Intellectual Property Rights Notice
              ****************************************

DEFINITIONS:
------------
For the purposes of this Intellectual Property Rights Notice
the terms ‘Perceptual Evaluation of Speech Quality Algorithm’
and ‘PESQ Algorithm’ refer to the objective speech quality
measurement algorithm defined in ITU-T Recommendation P.862;
the term ‘PESQ Software’ refers to the C-code component of P.862. 
These definitions also apply to those parts of ITU-T Recommendation 
P.862.2 and its associated source code that are common with P.862.

NOTICE:
-------
All copyright, trade marks, trade names, patents, know-how and
all or any other intellectual rights subsisting in or used in
connection with including all algorithms, documents and manuals
relating to the PESQ Algorithm and or PESQ Software are and remain
the sole property in law, ownership, regulations, treaties and
patent rights of the Owners identified below. The user may not
dispute or question the ownership of the PESQ Algorithm and
or PESQ Software.

OWNERS ARE:
-----------

1.      British Telecommunications plc (BT), all rights assigned
      to Psytechnics Limited
2.      Royal KPN NV, all rights assigned to OPTICOM GmbH

RESTRICTIONS:
-------------

The user cannot:

1.      alter, duplicate, modify, adapt, or translate in whole or in
      part any aspect of the PESQ Algorithm and or PESQ Software
2.      sell, hire, loan, distribute, dispose or put to any commercial
      use other than those permitted below in whole or in part any
      aspect of the PESQ Algorithm and or PESQ Software

PERMITTED USE:
--------------

The user may:

1.      Use the PESQ Software to:
      i)   understand the PESQ Algorithm; or
      ii)  evaluate the ability of the PESQ Algorithm to perform
           its intended function of predicting the speech quality
           of a system; or
      iii) evaluate the computational complexity of the PESQ Algorithm,
           with the limitation that none of said evaluations or its
           results shall be used for external commercial use.

2.      Use the PESQ Software to test if an implementation of the PESQ
      Algorithm conforms to ITU-T Recommendation P.862.

3.	With the prior written permission of both Psytechnics Limited
      and OPTICOM GmbH, use the PESQ Software in accordance with the
      above Restrictions to perform work that meets all of the following
      criteria:
      i)    the work must contribute directly to the maintenance of an
            existing ITU recommendation or the development of a new ITU
            recommendation under an approved ITU Study Item; and
      ii)   the work and its results must be fully described in a
            written contribution to the ITU that is presented at a formal
            ITU meeting within one year of the start of the work; and
      iii)  neither the work nor its results shall be put to any
            commercial use other than making said contribution to the ITU.
            Said permission will be provided on a case-by-case basis.


ANY OTHER USE OR APPLICATION OF THE PESQ SOFTWARE AND/OR THE PESQ
ALGORITHM WILL REQUIRE A PESQ LICENCE AGREEMENT, WHICH MAY BE OBTAINED
FROM EITHER OPTICOM GMBH OR PSYTECHNICS LIMITED. 

EACH COMPANY OFFERS OEM LICENSE AGREEMENTS, WHICH COMBINE OEM
IMPLEMENTATIONS OF THE PESQ ALGORITHM TOGETHER WITH A PESQ PATENT LICENSE
AGREEMENT. PESQ PATENT-ONLY LICENSE AGREEMENTS MAY BE OBTAINED FROM OPTICOM.


***********************************************************************
*  OPTICOM GmbH                    *  Psytechnics Limited             *
*  Naegelsbachstr. 38,             *  Fraser House, 23 Museum Street, *
*  D- 91052 Erlangen, Germany      *  Ipswich IP1 1HN, England        *
*  Phone: +49 (0) 9131 53020 0     *  Phone: +44 (0) 1473 261 800     *
*  Fax:   +49 (0) 9131 53020 20    *  Fax:   +44 (0) 1473 261 880     *
*  E-mail: info@opticom.de,        *  E-mail: info@psytechnics.com,   *
*  www.opticom.de                  *  www.psytechnics.com             *
***********************************************************************


Further information is also available from www.pesq.org


**************************************************************************



ITU-T NOTICE
------------

The ITU does not take any responsibility for difficulties or the 
impossibility of reading out the contents of these files. The ITU 
shall not be held liable for any direct, indirect, consequential or 
incidental damages arising out of the use of or inability to use the 
software.

The present notice must be included in any copy of the software or
the information contained therein.


ITU CONTACTS
------------

For distribution of update software, please contact:
Sales Department
ITU
Place des Nations
CH-1211 Geneve 20
SUISSE
email: sales@itu.int

For reporting problems, please contact TSB helpdesk service at:
TSB Helpdesk service
ITU
Place des Nations
CH-1211 Geneve 20
SUISSE
fax: +41 22 730 5853
email: tsbedh@itu.int


================
FILE DESCRIPTION
================

This electronic material is provided as part of ITU-T Rec. P.862 
Annex A. The package contains the following files:

README.TXT              This file

SOURCE CODE:
source\pesqmain.c       Main program
source\pesqdsp.c        PESQ DSP routines
source\pesqio.c         File I/O
source\dsp.c            Basic DSP routines
source\pesqmod.c        PESQ high-level model
source\dsp.h            Header file for dsp.h
source\pesq.h           General header file
source\pesqpar.h        Perceptual definitions

CONFORMANCE DATA:
conform\supp23_16k.txt  File pairs and PESQ scores for test 1(a)
conform\supp23_16k.bat  Batch script to assist with test 1(a)
conform\supp23_8k.txt   File pairs and PESQ scores for test 1(b)
conform\supp23_8k.bat   Batch script to assist with test 1(b)
conform\voipref_16k.txt File pairs and PESQ scores for test 2(a)
conform\voipref_16k.bat Batch script to assist with test 2(a)
conform\voipref_8k.txt  File pairs and PESQ scores for test 2(b)
conform\voipref_8k.bat  Batch script to assist with test 2(b)
conform\process.bat     Sample batch script to assist with preparing 
                          material for tests 1(b) and 2(a)
conform\supp23_wb.txt   File pairs and P.862.2 scores for conformance 
                          validation to Supplement 23 (wideband operation)
conform\supp23_wb.bat   Batch script to assist with validation of 
                          P.862.2 scores

conform\or105.wav       Conf. data, variable delay conditions
conform\or109.wav
conform\or114.wav
conform\or129.wav
conform\or134.wav
conform\or137.wav
conform\or145.wav
conform\or149.wav
conform\or152.wav
conform\or154.wav
conform\or155.wav
conform\or161.wav
conform\or164.wav
conform\or166.wav
conform\or170.wav
conform\or179.wav
conform\or221.wav
conform\or229.wav
conform\or246.wav
conform\or272.wav

conform\dg105.wav
conform\dg109.wav
conform\dg114.wav
conform\dg129.wav
conform\dg134.wav
conform\dg137.wav
conform\dg145.wav
conform\dg149.wav
conform\dg152.wav
conform\dg154.wav
conform\dg155.wav
conform\dg161.wav
conform\dg164.wav
conform\dg166.wav
conform\dg170.wav
conform\dg179.wav
conform\dg221.wav
conform\dg229.wav
conform\dg246.wav
conform\dg272.wav

conform\u_am1s01.wav
conform\u_am1s02.wav
conform\u_am1s03.wav
conform\u_am1s01b1c1.wav
conform\u_am1s01b1c7.wav
conform\u_am1s02b1c9.wav
conform\u_am1s01b1c15.wav
conform\u_am1s03b1c16.wav
conform\u_am1s03b1c18.wav
conform\u_am1s01b2c1.wav
conform\u_am1s02b2c4.wav
conform\u_am1s02b2c5.wav
conform\u_am1s03b2c5.wav
conform\u_am1s03b2c6.wav
conform\u_am1s03b2c7.wav
conform\u_am1s01b2c8.wav
conform\u_am1s03b2c11.wav
conform\u_am1s02b2c14.wav
conform\u_af1s01b2c16.wav
conform\u_af1s03b2c16.wav
conform\u_af1s02b2c17.wav
conform\u_af1s03b2c17.wav
conform\u_am1s03b2c18.wav
conform\u_af1s01.wav
conform\u_af1s02.wav
conform\u_af1s03.wav


