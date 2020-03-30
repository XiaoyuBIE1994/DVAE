# GNU Makefile for compiling PESQ version 2.0 sources (on the Linux 
# platform) available for download from ITU-T at the following 
# url: http://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en

# Originally wrote by Vineet Srivastava, JULY 2015.
# This file has been changed to fix some bugs by Xiaoyu BIE, MARS 2020

ARCH := $(shell uname -m)

ifeq ($(ARCH),i686)
ARCH_FLAG = -m32
else
ARCH_FLAG = -m64
endif

CFLAGS = ${ARCH_FLAG} -Wall -pipe -fPIC -funroll-loops -O3 -I.
LFLAGS = -lm

VPATH = .:source
BINDIR = bin
OBJDIR = obj
BIN = pesq

SRC = $(wildcard *.c)
OBJ = $(patsubst %.c, $(OBJDIR)/%.o, $(SRC))
	
all: prepare clean $(BINDIR)/$(BIN)

$(OBJDIR)/%.o: %.c
	@echo $(SRC)
	gcc -c $(CFLAGS) $< -o $@

prepare:
	@mkdir -p $(BINDIR)
	@mkdir -p $(OBJDIR)


.PHONY: clean
clean:
	@rm -f $(OBJDIR)/*.o
	@rm -f $(BINDIR)/$(BIN)

$(BINDIR)/$(BIN):$(BINDIR) $(OBJ)
	gcc $(OBJ) -o $@ $(LFLAGS)
	@echo ">>>>> PESQ compile finish"

# EOF
