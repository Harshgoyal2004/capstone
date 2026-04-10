%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This class provides the Vellore Institute of Technology - 
% School of Computer Science and Engineering Students 
% Thesis format. 
% This class was prepared by Dr. SUDHAKAR. P based on the 
% guidance from Dr. Kalyanaraman P 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{VITSCOPEThesis}[2026/02/13 VIT-Vellore SCOPE Thesis or Project report format]
\LoadClass[12pt]{report} % onehalfspacing

% Document Layout Settings
\RequirePackage[a4paper,left=1.5in,right=1in,top=1in,bottom=1in,nohead]{geometry}
\RequirePackage{setspace}
\setstretch{1.5}
% Declared for necessary formatting
\RequirePackage{mathptmx} % For times font
\RequirePackage{graphicx}

%Table of contents, List of figures, Tables 
\renewcommand*\contentsname{\MakeUppercase{Table of Contents}}
\renewcommand{\listtablename}{List of Tables}
\renewcommand{\listfigurename}{List of Figures}

% Chapter, section and subsection head formatting
\RequirePackage{titlesec}
% Chapter 
\titleformat{\chapter}[block]{\normalfont\Large\bfseries}{CHAPTER \thechapter}{0.3em}{\linebreak  \centering\MakeUppercase}
\titlespacing*{\chapter}{0pt}{-1em}{0pt}
% Section 
\titleformat{\section}[block]{\normalfont\bfseries}{\thesection}{0.3em}{\MakeUppercase}
% Subsection
\titleformat{\subsection}[block]{\normalfont\bfseries}{\thesubsection}{0.3em}{}

% other packages
\RequirePackage{subcaption}
\RequirePackage{natbib}
\RequirePackage{array}
\RequirePackage[sort]{acro}
\acsetup{
	list/template=tabular
}
\RequirePackage{longtable}
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{listings}
\RequirePackage{titlecaps}
\RequirePackage{pdfpages}
% other packages
\RequirePackage{listings}
\RequirePackage{float}
\RequirePackage{ragged2e}
\RequirePackage{ifthen}
\RequirePackage{rotating}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{xcolor}
\RequirePackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds}
\RequirePackage{caption}
\lstset{captionpos=b,
	frame=tb,
	basicstyle=\scriptsize\ttfamily,
	showstringspaces=false,
	keepspaces=true,
	language=Python}
% Bib formatting 
\bibliographystyle{apalike}

% Title Page Command
\newcommand{\makevittitlepage}{%
  \begin{titlepage}
    \centering
    \vspace*{0.2cm}
    \includegraphics[width=3cm]{images/vit_logo.png}\\[0.3cm]
    {\Large \textbf{VELLORE INSTITUTE OF TECHNOLOGY}}\\[0.2cm]
    {\large \textbf{SCHOOL OF COMPUTER SCIENCE AND ENGINEERING}}\\[0.2cm]
    {\large Vellore, Tamil Nadu -- 632014}
    
    \vfill
    
    \hrule height 2pt
    \vspace{0.3cm}
    {\LARGE \textbf{CAPSTONE PROJECT REPORT}}\\[0.2cm]
    \hrule height 2pt
    
    \vfill
    
    {\Large \textbf{UNIFIED MULTI-DISEASE PREDICTION SYSTEM\\[0.3cm]
    USING ADVANCED DEEP LEARNING \&\\[0.3cm]
    ENSEMBLE TECHNIQUES}}
    
    \vfill
    
    \begin{tabular}{ll}
      \textbf{Submitted by:} & Harsh Goyal \\
      \textbf{Registration Number:} & 22BCE2789 \\
      \textbf{Programme:} & Bachelor of Technology \\
      \textbf{Specialisation:} & Computer Science and Engineering \\
      \textbf{School:} & School of Computer Science and Engineering \\
    \end{tabular}
    
    \vspace{0.5cm}
    
    \begin{tabular}{ll}
      \textbf{Project Guide:} & Prof. Manikandan G \\
      \textbf{Designation:} & Assistant Professor, SCOPE \\
    \end{tabular}
    
    \vfill
    
    \hrule
    \vspace{0.3cm}
    {\large \textbf{Academic Year: 2025 -- 2026}}\\[0.2cm]
    {\large \textbf{Submitted in partial fulfilment of the requirements for}}\\[0.2cm]
    {\large \textbf{the Degree of Bachelor of Technology}}\\
    \vspace{0.3cm}
    \hrule
  \end{titlepage}
}

