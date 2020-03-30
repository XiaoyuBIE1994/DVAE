# GNU Makefile for compiling PESQ version 2.0 sources (on the Linux 
# platform) available for download from ITU-T at the following 
# url: http://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en

# Vineet Srivastava, JULY 2015.

ARCH := $(shell uname -m)

ifeq ($(ARCH),i686)
ARCH_FLAG = -m32
else
ARCH_FLAG = -m64
endif

CFLAGS = $(ARCH_FLAG) -Wall -pipe -fPIC -funroll-loops -O3 -I.
#LFLAGS = -L -lm
LFLAGS = -lm

SRC = $(*.c)
VPATH	= .:source
#SRC = $(source/*.c)
OBJ = $(SRC:.c=.o)
BIN = pesq
BINDIR = bin
OBJDIR = obj

OBJS_PESQ=\
	$(OBJDIR)/dsp.o \
	$(OBJDIR)/pesqdsp.o \
	$(OBJDIR)/pesqio.o \
	$(OBJDIR)/pesqmod.o \
	$(OBJDIR)/pesqmain.o \

binary=\
	$(BINDIR)/pesq

all: clean clean-bin $(binary) 

$(OBJDIR)/%.o: %.c
	gcc -c $(CFLAGS) $< -o $@

$(BINDIR)/$(BIN): 
	gcc $(OBJDIR)/$(OBJ) -o $@ $(LFLAGS)

prepare:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)
	
clean:
	rm -f $(OBJDIR)/*.o $(BINDIR)/*.so

clean-bin:
	rm -f $(BINDIR)/$(BIN)

$(BINDIR)/pesq:$(BINDIR) $(OBJS_PESQ)
	@echo "********************************************"
	@echo Making PESQ Model - Makfile given by Vineet.
	@echo Date: 8 JULY 2015
	@echo Binary name will be pesq
	@echo "********************************************"
	gcc $(OBJS_PESQ) -o $@ $(LFLAGS)

# EOF
