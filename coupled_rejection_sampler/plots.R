library(tidyverse)
library(arrow)
library(data.table)
library(here)

#Custom functions --------------------------------------------------------------
cs_theme <- function(){
  theme_bw()+
  theme(text = element_text(size=16), title = element_text(size=20),
        axis.text = element_text(size=16), legend.text = element_text(size=16),
        panel.background = element_rect(fill='#fafafa'),
        plot.background = element_rect(fill='#fafafa', color=NA),
        legend.background = element_rect(fill='#fafafa', color=NA),
        legend.key = element_rect(fill='#fafafa', color=NA),
        legend.box.background = element_rect(fill='#fafafa', color=NA))
}

cgg <- function(x){
  a<-deparse(substitute(x)) 
  path = paste0("results/plots/",a,".png")
  ggsave(path, x)
}

#Data import -------------------------------------------------------------------
path="results/gauss_tails/"
x_samples=read_feather(paste0(path, "x_samples.feather"))%>%transpose()
y_samples=read_feather(paste0(path, "y_samples.feather"))%>%transpose()
etas<-read_feather(paste0(path,"etas.feather"))
pxy<-read_feather(paste0(path,"pxy_list.feather"))
runtimes<-read_feather(paste0(path,"runtimes.feather"))
runtimes_cpu<-read_feather(paste0(path,"runtimes_cpu.feather"))

#DATA PREP----------------------------------------------------------------------
DATA1 <- data.frame(x= unlist(x_samples[10]), 
                    y = unlist(y_samples[10]))

DATA2 <- data.table(eta = unlist(etas), 
                    Actual = unlist(pxy[,1]), 
                    True = unlist(pxy[,2]), 
                    run_time=unlist(runtimes), 
                    run_time_cpu=unlist(runtimes_cpu))

#PLOTS -------------------------------------------------------------------------
plot1 <- DATA1%>%
  ggplot(aes(x=x, y=y))+
  geom_point(size=1, shape=16) +
  cs_theme()+
  labs(title="Marginal distributions coupling algorithm",
       subtitle="Algorithm 1 - Replication section 5.1 - 100k samples * 200",
       x="X~p", y="Y~q")

plot1 <- ggExtra::ggMarginal(plot1, type = "densigram")

plot1$layout$t[1] <- 1
plot1$layout$r[1] <- max(plot1$layout$r)

plot2 <- DATA2%>%
  ggplot()+
  geom_line(aes(x = eta, y = Actual, colour = "Actual"), linetype = "solid", linewidth=1.1) +
  geom_line(aes(x = eta, y = True, colour = "True"), linetype = "dashed", linewidth=1.1) +
  scale_y_log10() +
  scale_color_brewer(palette = "Set1")+
  labs(x = expression(eta), y = "Coupling Probability", 
       colour = "Legend") +
  cs_theme()+
  labs(title=paste0("Theoretical and empirical coupling as function of \u03B7"),
       subtitle="Algorithm 1 - Replication section 5.1 - 100k samples * 200",
       x="\u03B7", y="")+
  theme(legend.position = "bottom")


plot3 <- DATA2[2:199]%>%
  select(`RTX3060 (gpu, left)`=run_time, `5800X3D (cpu, right)`=run_time_cpu, eta)%>%
  mutate(`5800X3D (cpu, right)`=`5800X3D (cpu, right)`*1e-1)%>%
  pivot_longer(-eta)%>%
  ggplot(aes(x = eta, y = value, color=reorder(name, value))) +
  scale_y_continuous(
        name = "GPU Runtime",
        sec.axis = sec_axis(~.*10, name="CPU Runtime"))+
  geom_line(linewidth=1.1) +
  labs(x = expression(eta), y = "") +
  cs_theme()+
  scale_color_brewer(palette="Set1")+
  labs(title=paste0("Runtime in seconds as a function of \u03B7"),
       subtitle="Algorithm 1 - Replication section 5.1 - 100k samples * 200",
       x="\u03B7", y="", 
       color="Hardware")+
  theme(legend.position = "bottom")

#DIMENSION EXPERIMENT ----------------------------------------------------------
DIMEXP <- read_feather("results/gauss_dim/dim_1.feather")%>%
  mutate(dim=1)

for(i in 2:10){
  DIMEXP <-
    DIMEXP|>rbind(read_feather(paste0("results/gauss_dim/dim_",i,".feather"))%>%
    mutate(dim=i))
}

DIMEXP <- DIMEXP%>%
  pivot_longer(-dim)%>%
  group_by(dim, name)%>%
  summarise(mean=mean(value),
            lower=quantile(value, 0.025),
            upper=quantile(value, 0.975))


plot4 <- DIMEXP%>%filter(name=="sample_times")%>%
  ggplot(aes(x=as.integer(dim), y=mean, group=1))+
  geom_pointrange(aes(ymin = lower, ymax = upper)) +
  geom_line(linewidth=0.85)+
  labs(x = "Dimensions", y = "Runtime",
       title="Runtime as a function of the number of dimensions",
       subtitle="Algorithm 1 - N[0, I*2] and N[1, I*3] - 1k samples * 10D") +
  scale_x_continuous(breaks=1:10)+
  cs_theme()

plot5 <- DIMEXP%>%filter(name=="is_coupled")%>%
  ggplot(aes(x=dim, y=mean, group=1))+geom_line()+
  labs(x = "Dimensions", y = "Coupling probability",
       title="Coupling probability as function of the number of dimensions", 
       subtitle="Algorithm 1 - N[0, I*2] and N[1, I*3] - 1k samples * 10D")+
  scale_x_continuous(breaks=1:10)+
  cs_theme()

#TRIALS ---------------------------
TRIALS <- read_feather("results/gauss_dim/trials/dim_1.feather")%>%
  mutate(dim=1)%>%
  setDT()

for(i in 2:6){
  TRIALS <-
    TRIALS|>rbind(read_feather(paste0("results/gauss_dim/trials/dim_",i,".feather"))%>%
                    mutate(dim=i))
}


plot6 <- TRIALS%>%filter(n_trials<31)%>%
  mutate(dim=paste0("N° of Dimensions: ", dim))%>%
  ggplot(aes(x=n_trials, filll=dim, group=dim))+
  geom_histogram(binwidth = 1)+
  facet_wrap(~dim)+
  cs_theme()+
  labs(title="Number of trials before sucess",
       x="", y="n_trials", 
       subtitle="Algorithm 1 - N[0, I*2] and N[1, I*3] - 400 couplings * 6D")

#EXPORTS -----------------------------------------------------------------------
plot1|>cgg()
plot2|>cgg()
plot3|>cgg()
plot4|>cgg()
plot5|>cgg()
plot6|>cgg()


#BONUS CHI-SQUARE TEST ---------------------------------------------------------
chi_test <-function(ndim){
  data <- TRIALS[dim==ndim]$n_trials - 1
  
  p_estimate <- 1 / (mean(data)+1) 
  
  cutoff <- (TRIALS[dim==ndim]%>%
               group_by(n_trials)%>%
               summarise(n=n())%>%
               mutate(sum=spatstat.utils::revcumsum(n))%>%
               filter(sum>=6)%>%
               filter(sum==min(sum)))$n_trials-1
  
  data_grouped <- ifelse(data >= cutoff, paste0(cutoff, "+"), as.character(data))
  
  obs_freq <- table(factor(data_grouped, levels = c(0:(cutoff-1),paste0(cutoff, "+"))))
  prob <- sapply(0:(cutoff-1), function(x) dgeom(x, p_estimate))
  prob_plus <- 1 - sum(prob)
  exp_probs <- c(prob, prob_plus)
  exp_freq <- sum(obs_freq) * exp_probs
  chisq.test(x = obs_freq, p = exp_probs)
}

pval <- c()

for(i in 1:6){
  pval <- append(pval, chi_test(i)$p.value)
}

knitr::kable(data.frame(dim=1:6, pval), caption="Chi-Test adequation to geometric distribution")

TRIALS%>%group_by(dim)%>%
  summarise(p=1/mean(n_trials))%>%
  knitr::kable(caption = 
"Estimated probabilities of success ~ G(p)
Method of moments")



#THORISSON DATA
path="results/thorisson/"
c_coupling <- read_feather(paste0(path,"c_coupling.feather"))
runtimes <- read_feather(paste0(path,"run_times_exp1.feather"))
runtimes2 <- read_feather(paste0(path,"total_runtimes_exp2.feather"))%>%rename(`Total Runtime`=`0`)
dim_runtimes <- read_feather(paste0(path,"dim_runtimes.feather"))%>%rename(`Runtime (s)`=`0`)
dim_coupling <- read_feather(paste0(path,"dim_couplings.feather"))%>%rename(`Algorithm5`=`0`)
PXE <- read_feather(paste0(path,"P_X1_Y1_estimates.feather"))%>%rename(`Empirical Estimate`=`0`)
PXT <- read_feather(paste0(path,"P_X1_Y1_theoretical.feather"))%>%rename(`Theoretical Probability`=`0`)
Cc=seq(.01, 0.99, length.out=20)


#THORISSON PLOTS
plot7 <- c_coupling%>%
  mutate(prob=`0`, C=Cc)%>%
  ggplot(aes(x=C, y=prob))+
  geom_line()+
  labs(title="Probability of couplings as a function of C", 
       subtitle = "Algorithm 5 - Thorisson - N(0, I) and N([2,2], I*2) - 1k samples")+
  cs_theme()

plot8 <- runtimes%>%mutate(runtime=`0`,C=rep(Cc, each=10))%>%
  group_by(C)%>%
  summarise(mruntime=mean(runtime), 
            lower=quantile(runtime, 0.025), 
            upper=quantile(runtime, 0.975))%>%
  ggplot(aes(x=C, y=mruntime, group=1))+
  geom_pointrange(aes(ymin = lower, ymax = upper)) +
  geom_line(linewidth=0.85)+
  labs(x = "C", y = "Runtime",
       title="Runtime as a function of C",
       subtitle="Algorithm 5 - Thorisson - N(0, I) and N([2,2], I*2) - 500 samples * 10 * 20") +
  cs_theme()
  
plot9 <- data.frame(PXE, PXT, eta=seq(6, 6.59, 0.02))%>%
  pivot_longer(-eta)%>%
  ggplot(aes(x=eta, y=value, color=name))+
  geom_line(linewidth=1.1)+
  scale_color_brewer(palette="Set1")+
  cs_theme()+theme(legend.position = "bottom")+
  labs(title="Estimated and Theoretical P(X1 = Y1) for different values of a",
       subtitle = "Algorithm 5 - Replication section 5.1 - 1k samples",
       y="P(X1 = Y1)", color="")


plot10 <- runtimes2%>%mutate(eta=seq(6, 6.5, length.out=100))%>%
  ggplot(aes(x=eta, y=`Total Runtime`))+
  geom_line(linewidth=1.1)+
  labs(x = "eta", y = "Runtime",
       title="Runtime as a function of a",
       subtitle="Algorithm 5 - Thorisson - Replication section 5.1 - 1k samples * 100") +
  cs_theme()

plot11 <- data.frame(dim_coupling,DIMEXP%>%filter(name=="is_coupled")%>%
                       select("Algorithm1"=mean, dim))%>%
  pivot_longer(-dim)%>%
  ggplot(aes(x=dim, y=value, color=name))+
  geom_line(linewidth=1.1)+
  scale_color_brewer(palette="Set1")+
  scale_x_continuous(breaks=1:10)+
  cs_theme()+theme(strip.background = element_blank(),
                   strip.text = element_blank())+
  labs(x = "Dimensions", y = "",color="",
       title="Coupling probability as dimension function",
       subtitle="Algorithm 5 - N[0, I*2] and N[1, I*3] - 1-50k samples * 10D")+
  theme(legend.position = "bottom")

plot12 <- data.frame(dim_runtimes, dim=rep(1:10, each=50e3), type="Algorithm 5")%>%
  pivot_longer(-c(dim, type))%>%
  group_by(dim, type)%>%
  summarise(mean=mean(value),
            lower=quantile(value, 0.05),
            upper=quantile(value, 0.95))%>%
  rbind(DIMEXP%>%filter(name=="sample_times")%>%select(-name)%>%mutate(type="Algorithm 1"))%>%
  ggplot(aes(x=as.integer(dim), y=mean, color=type))+
  geom_pointrange(aes(ymin = lower, ymax = upper)) +
  scale_color_brewer(palette="Set1")+
  geom_line(linewidth=1.1)+
  labs(x = "Dimensions", y = "Runtime", color="",
       title="Runtime as a function of the number of dimensions",
       subtitle="Algorithm 1 - N[0, I*2] and N[1, I*3] - 1-50k samples * 10D") +
  scale_x_continuous(breaks=1:10)+
  cs_theme()+
  facet_wrap(~type, scales = "free", nrow=2)+
  theme(legend.position = "bottom")



#EXPORTS -----------------------------------------------------------------------
plot7|>cgg()
plot8|>cgg()
plot9|>cgg()
plot10|>cgg()
plot11|>cgg()
plot12|>cgg()

