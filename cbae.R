# Description: test idea of reconstructing both continuous and binary data with 
#     an autoencoder.
# Author: Haley Hunter-Zinck
# Date: August 6, 2020

# setup

tic = as.double(Sys.time())

library(keras)

# functions --------------------------------------------------

scale = function(x)
{
  min_value = min(x)
  max_value = max(x)
  s = (x - min_value) / (max_value - min_value)
  
  return(s)
}

unscale = function(s, min_value, max_value)
{
  return(s*(max_value - min_value) + min_value)
}

guess_variable_type = function(vec, variable_name, custom_categorical=c(),
                               custom_continuous=c(), custom_count=c(),
                               max_uniq=10, max_diff=1e-10)
{
  vec_clean = vec[which(!is.na(vec))]
  n_uniq = length(unique(vec_clean))
  
  # hard coded variable types
  if(is.element(variable_name, custom_categorical))
  {
    return("categorical")
  }
  if(is.element(variable_name, custom_continuous))
  {
    return("continuous")
  }
  if(is.element(variable_name, custom_count))
  {
    return("count")
  }
  
  # guess based on vector values
  if(n_uniq==1)
  {
    return("constant")
  }
  if(n_uniq==2)
  {
    return("binary")
  }
  if(n_uniq>max_uniq && (is.numeric(vec) || length(which(is.na(as.numeric(vec))))<length(vec)))
  {
    vec = as.double(vec)
    if(sum(abs(vec-round(vec))<=max_diff, na.rm=TRUE) == length(vec))
    {
      return("count")
    }
    return("continuous")
  }
  
  return("categorical")
}

gather_meta_data = function(x)
{
  header = c("type", "min", "max", "zero", "one")
  m = matrix(NA, nrow=ncol(x), ncol=length(header), dimnames=list(colnames(x), header))
  
  for(j in 1:ncol(x))
  {
    m[j,"type"] = guess_variable_type(vec=x[,j], variable_name=colnames(x)[j],
                                      custom_categorical=custom_categorical,
                                      custom_continuous=custom_continuous, custom_count=custom_count)
    
    if(m[j,"type"] == "binary")
    {
      m[j,c("one","zero")] = names(sort(table(x[,j])))
    } else if(m[j,"type"] == "constant")
    {
      m[j, "zero"] = unique(x[,j])
    } else if(is.element(m[j,"type"], c("continuous","count")))
    {
      x_j = as.double(x[,j])
      m[j,c("min","max")] = c(min(x_j), max(x_j))
    } 
  }
  
  return(m)
}

one_hot_encoding = function(vec, variable_name, delim="__")
{
  uval = sort(unique(vec), na.last = TRUE)
  
  # split categories
  col_labels = gsub(pattern = " ", replacement = "_", x = paste0(variable_name, delim, uval))
  mod = matrix(0, nrow = length(vec), ncol = length(uval), dimnames = list(c(),col_labels))
  for(j in 1:length(uval))
  {
    mod[which(vec == uval[j]),j] = 1
  }
  
  return(mod)
}

unravel_one_hot_encoding = function(mat, delim="__")
{
  vec = c()
  uval = unlist(lapply(strsplit(colnames(mat), split=delim),tail,n=1))
  for(i in 1:nrow(mat))
  {
    vec[i] = uval[which(mat[i,]==1)]
  }
  return(vec)
}

#' Take a generic matrix of multiple data types and discritize
#' in preparation fo synthetic data generation.
#' 
#' @param x Data matrix with samples as rows and features as column with column names specified.
#' @param m Meta-data matrix as output by gather_meta_data function
#' @return discretized version of x
discretize_matrix = function(x, m)
{
  s = c()
  
  for(j in 1:ncol(x))
  {
    type = m[j,"type"]
    variable_name = rownames(m)[j]
    s_j = c()
    
    if(type == "constant")
    {
      s_j = matrix(rep(0,nrow(x)), ncol = 1, dimnames = list(c(), variable_name))
     
    } else if(type == "continuous" || type == "count")
    {
      x_j = as.double(x[,j])
      s_j = matrix(scale(x_j), ncol=1, dimnames = list(c(), variable_name))
    } else if(type == "categorical")
    {
      s_j = one_hot_encoding(x[,j], variable_name = variable_name)
    } else if(type == "binary")
    {
      s_j = rep(0,nrow(x))
      s_j[which(x[,j] == m[j,"one"])] = 1
      s_j = matrix(s_j, ncol = 1,dimnames = list(c(), variable_name))
    } else
    {
      cat("Warning: variable type not recognized.  Appending as is.\n")
      s_j = x[,j,arr.ind=TRUE]
    }
    
    s = cbind(s, s_j)
  }
  
  return(s)
}

restore_matrix = function(s, m, delim="__")
{
  x_prime = c()
  variable_names = unlist(lapply(strsplit(colnames(s), split=delim), head, n=1))
  
  for(j in 1:nrow(m))
  {
    type = m[j,"type"]
    variable_name = rownames(m)[j]
    s_j = s[,which(variable_names==rownames(m)[j])]
    
    if(type == "constant")
    {
      x_j = rep(m[j,"zero"],nrow(x_prime))
      
      idx_nonzero = which(s_j != 0)
      x_j[idx_nonzero] = NA
      
    } else if(type == "continuous" || type == "count")
    {
      s_j = as.double(s_j)
      min_value = as.double(m[j,"min"])
      max_value = as.double(m[j,"max"])
      x_j = unscale(s=s_j, min_value=min_value, max_value=max_value)
      
      if(type == "count")
      {
        x_j = round(x_j)
      }
    } else if(type == "binary")
    {
      x_j = rep(NA,length(s_j))
      x_j[which(s_j==0)]=m[j,"zero"]
      x_j[which(s_j==1)]=m[j,"one"]
    } else if(type == "categorical")
    {
      x_j = unravel_one_hot_encoding(s_j, delim=delim)
    } else
    {
      cat("Warning: variable type not recognized.  Appending as is.\n")
      x_j = s_j
    }
  
    x_prime = cbind(x_prime, x_j)
  }
  
  colnames(x_prime) = rownames(m)
  return(x_prime)
}

check_restore = function(x, x_prime, m, threshold=0.05)
{
  msg_pass = "PASS"
  msg_fail = "FAIL"
  
  # dimensions
  cat("Checking dimensions...")
  if(identical(dim(x), dim(x_prime)))
  {
    cat(msg_pass,"\n")
  } else
  {
    cat(msg_fail,"\n")
  }
  
  # column names
  cat("Checking column names...")
  if(identical(colnames(x),colnames(x_prime)))
  {
    cat(msg_pass,"\n")
  } else
  {
    cat(msg_fail,"\n")
  }
  
  # for each column
  for(j in 1:ncol(x))
  {
    cat("Checking column '", colnames(x)[j], "' (", j, "/", ncol(x), ")...", sep="")
    
    type = m[j,"type"]
    variable_name = rownames(m)[j]
    x_j = x[,j]
    xp_j = x_prime[,j]
    
    if(type == "constant")
    {
      if(length(unique(x_j)) == length(unique(xp_j))
          && unique(x_j) == unique(xp_j))
      {
        cat(msg_pass,"\n")
      } else
      {
        cat(msg_fail,"\n")
      }
      
    } else if(type == "continuous" || type == "count")
    {
      x_j = as.double(x_j)
      xp_j = as.double(xp_j)
      if(wilcox.test(tmp1,tmp2)$p.value > threshold)
      {
        cat(msg_pass,"\n")
      } else
      {
        cat(msg_fail,"\n")
      }
      
    } else if(type == "categorical")
    {
      if(identical(sort(unique(x_j)), sort(unique(xp_j))))
      {
        cat(msg_pass,"\n")
      } else
      {
        cat(msg_fail,"\n")
      }
      
    } else if(type == "binary")
    {
      if(length(unique(x_j))==length(unique(xp_j))
         && identical(sort(unique(x_j)), sort(unique(xp_j))))
      {
        cat(msg_pass,"\n")
      } else
      {
        cat(msg_fail,"\n")
      }
      
    } else
    {
      cat(msg_fail,"\n")
    }
  }
}

train_ae = function(x, n_epoch=5)
{
  library(keras)
  
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 256, activation = 'tanh', input_shape = ncol(x),
                kernel_regularizer = regularizer_l2(l = 0.001)) %>%
    layer_dense(units = ncol(x), activation = 'sigmoid') %>%
    compile(
      optimizer = 'adam',
      loss = 'mean_squared_error',
      metrics = c('accuracy'),
      initializer = initializer_glorot_normal()
    )
  
  model %>% fit(x=x, y=x, epochs = n_epoch)
  
  return(model)
}

predict_keras = function(x, model)
{
  library(keras)
  
  x_prime = model %>% predict(x)
  colnames(x_prime) = colnames(x)
  
  return(x_prime)
}

indicator_max = function(vec)
{
  idx = which(vec==max(vec))
  if(length(idx)>1)
  {
    idx = sample(idx,1)
  }
  
  return(as.double(1:length(vec)==idx))
  
}

process_ae_output = function(s, m, delim="__")
{
  s_post = c()
  variable_names = unlist(lapply(strsplit(colnames(s), split=delim), head, n=1))
  
  for(j in 1:nrow(m))
  {
    type = m[j,"type"]
    variable_name = rownames(m)[j]
    s_j = s[,which(variable_names==rownames(m)[j])]
    
    if(type == "constant" || type == "binary")
    {
      x_j = round(s_j)
    } else if(type == "continuous" || type == "count")
    {
      x_j = s_j
    } else if(type == "categorical")
    {
      x_j = t(apply(s_j,1,indicator_max))
    } else
    {
      cat("Warning: variable type not recognized.  Appending as is.\n")
      x_j = s_j
    }
    
    s_post = cbind(s_post, x_j)
  }
  
  colnames(s_post) = colnames(s)
  return(s_post)
}

#' Use standard R plotting functions to plot multiple density curves.
#' @param my_list List of numeric vectors 
#' @param labels Text label for each element of my_list
#' @param xlab Label for the x-axis
#' @param main Label for the title
#' @param lwd Width of the density curves
#' @param colors Vector of R color values for each element of my_list
#' @param legend_pos Position of the legend
#' @return TRUE
#' @example 
#' plot_list_as_density(my_list=list(d1=rnorm(100), d2=rnorm(100,mean=5)), labels=c("density 1", "density 2"), xlab="Value")
plot_list_as_density = function(my_list, labels, xlab, main = "", 
                                lwd = 3, colors = NULL, legend_pos = "topright",
                                pal="Set2")
{
  library(RColorBrewer)
  
  if(is.null(colors))
  {
    if(length(my_list)<3)
    {
      colors = brewer.pal(n = 3, name = pal)[1:length(my_list)]
    } else
    {
      colors = brewer.pal(n = length(my_list), name = pal)
    }
  }
  
  # y-axis limits
  y_max = -Inf
  for(i in 1:length(my_list))
  {
    y_max_i = max(density(my_list[[i]], na.rm=TRUE)$y)
    if(y_max_i > y_max)
    {
      y_max = y_max_i
    }
  }
  y_limits = c(0, y_max)
  
  # x-axis limits
  x_max = -Inf
  x_min = Inf
  for(i in 1:length(my_list))
  {
    x_max_i = max(density(my_list[[i]], na.rm=TRUE)$x)
    if(x_max_i > x_max)
    {
      x_max = x_max_i
    }
    
    x_min_i = min(density(my_list[[i]], na.rm=TRUE)$x)
    if(x_min_i < x_min)
    {
      x_min = x_min_i
    }
  }
  x_limits = c(x_min, x_max)
  
  plot(density(my_list[[1]], na.rm=TRUE), ylim = y_limits, xlim = x_limits,
       xlab = xlab, ylab = "Density", col = colors[1], main = main, lwd = lwd)
  
  if(length(my_list) > 1)
  {
    for(i in 2:length(my_list))
    {
      points(density(my_list[[i]], na.rm=TRUE)$x, density(my_list[[i]], na.rm=TRUE)$y, type = "l", col = colors[i], lwd = lwd)
    }
  }
  legend(x = legend_pos, legend = labels, lwd = lwd, col = colors)
  
  return(T)
}


# main ----------------

# create demo dataset
n = 1e2
v1 = sample(c("A","B","C"),n,replace=T)
v2 = rnorm(n)
v3 = rep("elephant", n)
v4 = sample(c("hello","world"),n,replace=T)
x = cbind(v1,v2,v3,v4)
m = gather_meta_data(x=x)
s = discretize_matrix(x,m)

# check trivial restoration
x_prime = restore_matrix(s,m)
head(x)
head(x_prime)
check_restore(x,x_prime,m)

# check autoencoder restoration
model = train_ae(s, n_epoch=1e3)
s_ae = predict_keras(x=s, model=model)
p_ae = process_ae_output(s_ae, m)
x_ae = restore_matrix(p_ae,m)
head(x)
head(x_ae)
check_restore(x,x_ae,m)

par(mfrow=c(1,1))
plot_list_as_density(my_list=list(as.double(x[,"v2"]),as.double(x_ae[,"v2"])), labels=c("Raw","AE"), xlab="V2")

# close out -----------

toc = as.double(Sys.time())
cat("Runtime: ", toc - tic, " s\n", sep="")