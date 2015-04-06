### Configurable options (in case client Makefile is not including them)
# Root project file(s) (without .tex)
MAIN?=
# Default output format [ps|pdf]
DEFAULT?=pdf
# Generate pdf from ps (either use pdflatex)
PS2PDF?=n
# Check syntax before building
PRE_CHECK?=y
# Extra options for rubber
REXTRAOPTS?=
# Extra directories to "make"
PRE_EXTRADIRS?=
POST_EXTRADIRS?=
# Extra objectives to "make" (need <obj>, <obj>-clean and <obj>-distclean)
PRE_EXTRAOBJS?=
POST_EXTRAOBJS?=



####################################
### No need to edit from here on ###
####################################

### Tarball constructs
DATE=$(shell date +%Y%m%d%H%M)
DIR=$(shell basename ${CURDIR})
BASE=$(subst -2.*,,${DIR})
DIRNEW=${BASE}-${DATE}


### Other opts
# - none -


### Programs
TPLDIR=$(CURDIR)/$(dir $(lastword $(MAKEFILE_LIST)))
PATH:=$(TPLDIR)/extra/:$(PATH)
RUBBER=rubber
RUBBERINFO=rubber-info
ROPTS=$(REXTRAOPTS) -Wall -f -I . 
CHECK=lacheck
SVN=svn


### Generic rules

all: $(addsuffix .$(DEFAULT),$(MAIN))

check-RUBBER:
	@if test ! -x "`which $(RUBBER)`"; then \
		echo "Program \`$(RUBBER)' is not installed"; \
		exit 1; \
	fi

check-CHECK:
	@if test "$(PRE_CHECK)" = "y" -a ! -x "`which $(CHECK)`"; then \
		echo "Program \`$(CHECK)' is not installed"; \
		exit 1; \
	fi

check-SVN:
	@if test ! -x "`which $(SVN)`"; then \
		echo "Program \`$(SVN)' is not installed"; \
		exit 1; \
	fi

ps: $(addsuffix .ps,$(MAIN))

pdf: $(addsuffix .pdf,$(MAIN))

.PHONY: force pre-extraobjs

force:
	@#f

pre-extraobjs: $(PRE_EXTRAOBJS)

%.ps %.pdf: check-RUBBER check-CHECK pre-extraobjs %.tex
	@for i in $(PRE_EXTRADIRS); do \
		echo "make -C $$i"; \
		make -C $$i; \
	done
	@SUF=`echo $@ | sed -e 's/.*\.//'`; \
	if [ "$(PS2PDF)" = "y" -a $$SUF = "pdf" ] || [ -n "`$(RUBBERINFO) --deps $* | grep $(TPLDIR)tex/myslides.cls`" ];then \
		cmd="$(RUBBER) $(ROPTS) --ps --pdf $*"; \
	else \
		cmd="$(RUBBER) $(ROPTS) --$$SUF $*"; \
	fi; \
	echo "$$cmd"; \
	[ "$(PRE_CHECK)" = "n" ] || $(CHECK) $*; \
	$$cmd
	@for i in $(POST_EXTRADIRS); do \
		echo "make -C $$i"; \
		make -C $$i; \
	done
	@[ -z "$(POST_EXTRAOBJS)" ] || make $(POST_EXTRAOBJS)


help:
	@echo "Available objectives:"
	@echo "   all        : generate all documents in default format ($(DEFAULT))"
	@echo "   ps         : generate all postscript documents"
	@echo "   pdf        : generate all pdf documents"
	@echo "   <main>.ps  : generate a single postscript document"
	@echo "   <main>.pdf : generate a single pdf document"
	@echo ""
	@echo "   check      : check syntax correctness in documents"
	@echo "   list       : list all available documents"
	@echo ""
	@echo "   clean      : clean intermediate files"
	@echo "   distclean  : clean all files"
	@echo "   dist       : create a tarball of this project (all files inside a directory)"
	@echo "   dist-here  : create a tarball of this project"
	@echo ""
	@echo "   dtx        : rebuild LaTeX sources and documentation"

check: check-CHECK
	@for f in $(MAIN); do \
        $(CHECK) $$f; \
    done

list:
	@echo "$(MAIN)"

clean: check-RUBBER
	@[ -z "$(POST_EXTRAOBJS)" ] || make $(addsuffix -clean,$(POST_EXTRAOBJS))
	@for main in $(MAIN); do \
		$(RUBBER) --clean $$main; \
		figs=`figures-clean $$main ps pdf | tr \\\n " "`; \
		if [ -n "$$figs" ]; then \
			echo "rm -f $$figs"; \
			rm -f $$figs; \
		fi; \
	done
	rm -f *.nav *.out *.snm *.vrb *.idx *.brf
	@for i in $(PRE_EXTRADIRS) $(POST_EXTRADIRS); do \
		echo "make -C $$i $@"; \
		make -C $$i $@; \
	done
	@[ -z "$(PRE_EXTRAOBJS)" ] || make $(addsuffix -clean,$(PRE_EXTRAOBJS))

distclean: clean
	@[ -z "$(POST_EXTRAOBJS)" ] || make $(addsuffix -distclean,$(POST_EXTRAOBJS))
	@for i in $(MAIN); do \
		echo "rm -f $$i.pdf $$i.ps"; \
		rm -f $$i.pdf $$i.ps; \
	done
	@for i in $(PRE_EXTRADIRS) $(POST_EXTRADIRS); do \
		echo "make -C $$i $@"; \
		make -C $$i $@; \
	done
	@[ -z "$(PRE_EXTRAOBJS)" ] || make $(addsuffix -distclean,$(PRE_EXTRAOBJS))

dist: clean
	cp -r ../$(DIR) ../$(DIRNEW)
	tar zcf ../$(DIRNEW).tgz ../$(DIRNEW)
	rm -Rf ../$(DIRNEW)

dist-here: clean
	tar zcf $(DIRNEW).tgz .

dtx:
	@rm -f $(TPLDIR)/templates.* $(TPLDIR)/tex/*
	@(cd $(TPLDIR) && pdflatex src/templates.ins)
	@(cd $(TPLDIR) && rubber --pdf src/templates.dtx)
	@rm -f `find templates/ -maxdepth 1 -mindepth 1 -type f -name templates.* -not -name templates.pdf`
