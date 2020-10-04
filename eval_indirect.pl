#!/usr/bin/perl

##########################################
#This script is originally created to evaluate Semeval-2016 Task 6:
#Detecting Stance in Tweets
#http://alt.qcri.org/semeval2016/task6/
#
#Author: Xiaodan Zhu (www.xiaodanzhu.com)
#
#Created date: Oct. 1, 2015
#
#This simple script is free but if you choose to modify it
#please include the description above.
##########################################

##########################################
#For usage, type perl eval.pl -u
##########################################

use strict;

if(@ARGV == 1 && $ARGV[0] eq "-u"){
	printUsage();
	die "\n";
}

if(@ARGV != 4){
	print STDERR "\nError: Number of parameters are incorrect!\n";
	printUsage();
	die "\n";
}

my $fnGold = $ARGV[0];
open(IN, $fnGold) || die "Error: Cannot open the gold-standard file.\n";
my @goldLns = <IN>;
chomp @goldLns;
close(IN);

my $fnGuess = $ARGV[1];
open(IN, $fnGuess) || die "Error: Cannot open the file containing your prediction.\n";
my @guessLns = <IN>;
chomp @guessLns;
close(IN);

my $fnDirectIdList = $ARGV[2];
open(IN, $fnDirectIdList) || die "Error: Cannot open the file containing the manual marks for direct or indirect tagets.\n";
my @directIdList = <IN>;
chomp @directIdList;
close(IN);

my $fnDirectToCompute = $ARGV[3];

if(@guessLns != @goldLns){
	print STDERR "\nError: make sure the number of lines in your prediction file is same as that in the gold-standard file!\n";
	print STDERR sprintf("The gold-standard file contains %d lines, but the prediction file contains %d lines.\n", scalar(@goldLns), scalar(@guessLns));
	die "\n";
}

my @cats = ("FAVOR", "AGAINST", "NONE", "UNKNOWN");
my @tars = ("Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion");
#my @tarsB = ("Donald Trump");

my %catsHash = map{$_ => 1}@cats;
my %numOfTruePosOfEachCat = ();
my %numOfGuessOfEachCat = ();
my %numOfGoldOfEachCat = ();

my %numOfTruePosOfCatTar = {};
my %numOfGuessOfCatTar = {};
my %numOfGoldOfCatTar = {};
my $nTotalTweetsUsed = 0;

foreach my $cat (@cats) {
	foreach my $tar (@tars) {
		$numOfTruePosOfCatTar{$cat}{$tar} = 0;
		$numOfGuessOfCatTar{$cat}{$tar} = 0;
		$numOfGoldOfCatTar{$cat}{$tar} = 0;
	}
}


for (my $i = 0; $i < @guessLns; $i+=1) {

	my $guessLn = $guessLns[$i];
	$guessLn =~ s/\r//g;
	my $goldLn = $goldLns[$i];
	$goldLn =~ s/\r//g;

	if($goldLn eq "ID	Target	Tweet	Stance"){
		next;
	}

	my $directId = @directIdList[$i];
	$directId = substr($directId, 0, 1);
	if(($fnDirectToCompute ne "all") && ($fnDirectToCompute ne $directId)){
		next;
	}

	my @goldArr = split(/\t/, $goldLn);
	if(@goldArr != 4){
		print STDERR sprintf("\nError: the following line in the gold-standard file does not have a correct format:\n\n%s\n\n",$goldLn);
		print STDERR "Correct format: ID<Tab>Target<Tab>Tweet<Tab>Stance\n";
		die "\n";
	}

	my @guessArr = split(/\t/, $guessLn);
	if(@guessArr != 4){
		print STDERR sprintf("\nError: the following line in your prediction file does not have a correct format:\n\n%s\n\n",$guessLn);
		print STDERR "Correct format: ID<Tab>Target<Tab>Tweet<Tab>Stance";
		die "\n";
	}

	my $guessLbl = $guessArr[3];
	my $goldLbl = $goldArr[3];

	if(!defined($catsHash{$goldLbl})){
		print STDERR sprintf("\nError: the stance label \"%s\" in the following line of the gold-standard file is invalid:\n\n%s\n\n",$goldLbl, $goldLn);
		print STDERR "Correct labels in gold-standard file can be: FAVOR, AGAINST, NONE, or UNKNOWN (case sensitive). \n";
		die "\n";
	}

	if(!defined($catsHash{$guessLbl})){
		print STDERR sprintf("\nError: the stance label \"%s\" in the following line of the prediction file is invalid:\n\n%s\n\n",$guessLbl, $guessLn);
		print STDERR "Correct labels in predication file can be: FAVOR, AGAINST, NONE, or UNKNOWN (case sensitive). \n";
		die "\n";
	}

	my $tar = $goldArr[1];

    $nTotalTweetsUsed += 1;

	$numOfGoldOfEachCat{$goldLbl} += 1;
	$numOfGuessOfEachCat{$guessLbl} += 1;
	if($guessLbl eq $goldLbl){
		$numOfTruePosOfEachCat{$guessLbl} += 1;
	}

	if(($tar eq "Climate Change is a Real Concern") && ($guessLbl eq "AGAINST")){
		my $debugtmp = 1;
	}

	$numOfGoldOfCatTar{$goldLbl}{$tar} += 1;
	$numOfGuessOfCatTar{$guessLbl}{$tar} += 1;
	if($guessLbl eq $goldLbl){
		$numOfTruePosOfCatTar{$guessLbl}{$tar} += 1;
	}
}

#compute precision, recall, and f-score
my %precByCat = ();
my %recallByCat = ();
my %nTpByCat = ();
my %nGuessByCat = ();
my %nGoldByCat = ();
my %fByCat = ();

my %precByCatTar = {};
my %recallByCatTar = {};
my %nTpByCatTar = {};
my %nGuessByCatTar = {};
my %nGoldByCatTar = {};
my %fByCatTar = {};

my %precByTarCat = {};
my %recallByTarCat = {};
my %nTpByCatTarCat = {};
my %nGuessByTarCat = {};
my %nGoldByTarCat = {};
my %fByCatTarCat = {};

my $macroF = 0.0;

foreach my $cat (@cats) {
	my $nTp = $numOfTruePosOfEachCat{$cat};
	my $nGuess = $numOfGuessOfEachCat{$cat};
	my $nGold = $numOfGoldOfEachCat{$cat};

	my $p = 0;
	my $r = 0;
	my $f = 0;

	$p = $nTp/$nGuess if($nGuess != 0);
	$r = $nTp/$nGold if($nGold != 0);
	$f = 2*$p*$r/($p+$r) if($p + $r != 0);

	$nTpByCat{$cat} = $nTp;
	$nGuessByCat{$cat} = $nGuess;
	$nGoldByCat{$cat} = $nGold;

	$precByCat{$cat} = $p;
	$recallByCat{$cat} = $r;
	$fByCat{$cat} = $f;

	###############
	#Cat-target
	###############
	foreach my $tar (@tars) {
		my $nTpTmp = $numOfTruePosOfCatTar{$cat}{$tar};
		my $nGuessTmp = $numOfGuessOfCatTar{$cat}{$tar};
		my $nGoldTmp = $numOfGoldOfCatTar{$cat}{$tar};

		my $pTmp = 0;
		my $rTmp = 0;
		my $fTmp = 0;

		$pTmp = $nTpTmp/$nGuessTmp if($nGuessTmp != 0);
		$rTmp = $nTpTmp/$nGoldTmp if($nGoldTmp != 0);
		$fTmp = 2*$pTmp*$rTmp/($pTmp+$rTmp) if($pTmp + $rTmp != 0);

		$nTpByCatTar{$cat}{$tar} = $nTpTmp;
		$nGuessByCatTar{$cat}{$tar} = $nGuessTmp;
		$nGoldByCatTar{$cat}{$tar} = $nGoldTmp;
		
		$precByCatTar{$cat}{$tar} = $pTmp;
		$recallByCatTar{$cat}{$tar} = $rTmp;
		$fByCatTar{$cat}{$tar} = $fTmp;
	}
}

#print results
#=for comment
my $macroF = 0.0;
print STDOUT sprintf("\n\n============\n");
print STDOUT sprintf("Results: Macro over category\n");
print STDOUT sprintf("============\n");
my $nCat = 0;
foreach my $cat (@cats) {
	if($cat eq "FAVOR" || $cat eq "AGAINST"){
		$nCat += 1;
		$macroF += $fByCat{$cat};
		print STDOUT sprintf("%-9s precision: %.4f recall: %.4f f-score: %.4f\n", $cat, $precByCat{$cat}, $recallByCat{$cat}, $fByCat{$cat});
#		print STDOUT sprintf("%-9s precision: %.4f recall: %.4f f-score: %.4f TruePos: %d nGuess: %d nGold: %d\n", $cat, $precByCat{$cat}, $recallByCat{$cat}, $fByCat{$cat}, $nTpByCat{$cat}, $nGuessByCat{$cat}, $nGoldByCat{$cat});
	}
}
$macroF = $macroF/$nCat;
print STDOUT sprintf("------------\n");
print STDOUT sprintf("Macro F: %.4f\n\n", $macroF);
#print STDOUT sprintf("Total Tweets number used: %d\n\n", $nTotalTweetsUsed);
#=cut

=for comment
#print results
my $macroF = 0.0;
#print STDOUT sprintf("\n\n============\n");
#print STDOUT sprintf("Results: Macro over target then over category\n");
#print STDOUT sprintf("============\n");
my $nCat = 0;
foreach my $cat (@cats) {
	if($cat eq "FAVOR" || $cat eq "AGAINST"){
		$nCat += 1;
#		print STDOUT sprintf("%-9s precision: %.4f recall: %.4f f-score: %.4f TruePos: %d nGuess: %d nGold: %d\n", $cat, $precByCat{$cat}, $recallByCat{$cat}, $fByCat{$cat}, $nTpByCat{$cat}, $nGuessByCat{$cat}, $nGoldByCat{$cat});
		my $avrFOverTar = 0;
		foreach my $tar (@tars) {
#			print STDOUT sprintf("%-60s precision: %.4f recall: %.4f f-score: %.4f TruePos: %d nGuess: %d nGold: %d\n", $tar, $precByCatTar{$cat}{$tar}, $recallByCatTar{$cat}{$tar}, $fByCatTar{$cat}{$tar}, $nTpByCatTar{$cat}{$tar}, $nGuessByCatTar{$cat}{$tar}, $nGoldByCatTar{$cat}{$tar});
			$avrFOverTar += $fByCatTar{$cat}{$tar};

		}
#		print STDOUT sprintf("macro-f-score over targets: %.4f", $avrFOverTar/(scalar @tars));
#		print STDOUT "\n";
		$macroF += $avrFOverTar/(scalar @tars);
	}
}
$macroF = $macroF/$nCat;
#print STDOUT sprintf("------------\n");
#print STDOUT sprintf("Macro F: %.4f\n\n", $macroF);


#print results
my $macroF = 0.0;
print STDOUT sprintf("\n\n============\n");
print STDOUT sprintf("Results: Macro over category then over targets\n");
print STDOUT sprintf("============\n");
foreach my $tar (@tars) {
		print STDOUT sprintf("\n-----%s-----\n", $tar);
		my $nCat = 0;
		my $avrFOverCat = 0;
		foreach my $cat (@cats) {
			if($cat eq "FAVOR" || $cat eq "AGAINST"){
				$nCat += 1;
				print STDOUT sprintf("%-9s precision: %.4f recall: %.4f f-score: %.4f TruePos: %d nGuess: %d nGold: %d\n", $cat, $precByCatTar{$cat}{$tar}, $recallByCatTar{$cat}{$tar}, $fByCatTar{$cat}{$tar}, $nTpByCatTar{$cat}{$tar}, $nGuessByCatTar{$cat}{$tar}, $nGoldByCatTar{$cat}{$tar});
				$avrFOverCat += $fByCatTar{$cat}{$tar};
			}
		}
		print STDOUT sprintf("target f-score: %.4f", $avrFOverCat/($nCat));
		print STDOUT "\n";
		$macroF += $avrFOverCat/($nCat);
}
$macroF = $macroF/(scalar @tars);
print STDOUT sprintf("------------\n");
print STDOUT sprintf("Macro f-score of this team: %.4f\n\n", $macroF);
=cut

sub printUsage {
	print STDERR "\n---------------------------\n";
	print STDERR "Usage:\nperl eval.pl goldFile guessFile directOrIndirectTargetAnnotationFile id\n\n";
	print STDERR "goldFile:  file containing gold standards;\nguessFile: file containing your prediction;\n";
	print STDERR "directOrIndirectTargetAnnotationFile:  file containing manual annotations that indicates direct or indirect targets;\n";
	print STDERR "id (take the value 1 or 2):\n  1--calculate F-score of your system on the subset of tweets with direct tagets.\n  2--calculate F-score of your system on the subset of tweets with indirect tagets.\n";
	print STDERR "---------------------------\n";
}
