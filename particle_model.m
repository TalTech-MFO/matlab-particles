clear;

%-----------------
% Load data
data_path = "./data";
% files = dir(data_path+"/surface*.nc");
% [uu, vv, lon, lat, dates] = load_data(files, data_path, "uu", "vv", "lonc", "latc");

files = dir(data_path+"/cmems_feb*.nc");
[uu, vv, lon, lat, dates] = load_data(files, data_path, "uo", "vo", "longitude", "latitude");

%-----------------
% Simulation setup
current_time = dates(1);
time_step = 600.;

n_particles = 500;

% Initial positions
rng(123, 'simdTwister')
dx_init = rand(n_particles, 1) / 100.;
dy_init = rand(n_particles, 1) / 100.;

px = 24.63 + dx_init;
py = 59.66 + dy_init;

px_init = px;
py_init = py;

% Define the solver
solver = @(px, py, current_time) euler_step(px, py, current_time, time_step, uu, vv, lon, lat, dates);

%-----------------
% Create plot
water_vel = sqrt((uu .* uu) + (vv .* vv));
h_water_vel = pcolor(lon, lat, water_vel(:,:,1)');
colormap("jet");
caxis([0.,0.6]);
h_water_vel.EdgeColor = "none";
h_water_vel.FaceColor = 'interp';
hold on;
colorbar();
view(0,90);
% Create scatter objects with initial data
h_track = scatter(px, py, 10, [0.5 0.5 0.5], 'filled'); 
h_current = scatter(px, py, 30, 'red', 'filled');
drawnow;

% Initialize track arrays
save_step = 5
buffer_maxsize = 40;
px_all = px;
py_all = py;

%-----------------
% Run simulation
istep = 0;
while current_time < dates(end)  
    
    istep = istep + 1;
    
    if mod(istep, save_step) == 0
        % Append current positions to the track arrays
        buffer_start = max(numel(px_all) - (buffer_maxsize*n_particles), 1);
        px_all = [px_all(buffer_start:end); px];
        py_all = [py_all(buffer_start:end); py];

    end
   
    [px, py] = solver(px, py, current_time);
   
    if mod(istep, save_step) == 0
        % Update scatter data
        time_idx = floor(get_index(current_time, dates));
        set(h_water_vel, 'CData', water_vel(:,:,time_idx)');
        set(h_track, 'XData', px_all, 'YData', py_all);
        set(h_current, 'XData', px, 'YData', py);
        
        drawnow;
    end
    
    current_time = current_time + seconds(time_step);
end

disp("Done");

%-----------------
% Plot initial and final positions
hold off;
figure();
h_water_vel = pcolor(lon, lat, water_vel(:,:,time_idx)');
colormap("jet");
caxis([0.,0.6]);
h_water_vel.EdgeColor = "none";
h_water_vel.FaceColor = 'interp';
hold on;
scatter(px_init, py_init, 20, "green", "filled");
scatter(px, py, 20, "red", "filled");

xlim([22.9, 27]);
ylim([59, 60.2]);

fname = "fig_result_" + string(datetime('now','TimeZone','local','Format','HHmmss')) + ".png";
f = gcf;
exportgraphics(f, fname, "Resolution", 300);

%==========================================================================

function [uu, vv, lon, lat, dates] = load_data(files, data_path, uu_name, vv_name, lon_name, lat_name)
    
    num_files = length(files);
    
    uu = [];
    vv = [];
    lon = [];
    lat = [];
    dates = [];

    for ifile = 1:num_files
        full_path = data_path + "/" + files(ifile).name;
        
        if ifile == 1
            lon = squeeze(ncread(full_path, lon_name));
            lat = squeeze(ncread(full_path, lat_name));
        end 
        
        uu = cat(3, uu, squeeze(ncread(full_path, uu_name)));
        vv = cat(3, vv, squeeze(ncread(full_path, vv_name)));
        
        time = squeeze(ncread(full_path, "time"));
        try
            disp("Reading time attribute");
            time_units = string(ncreadatt(full_path, "time", "units")).char;
        catch
            warning(" Could not read time units with ncread. Trying h5");
            time_units = string(h5readatt(full_path, "/time", "units")).char;
        end
        epoch = datetime(time_units(15:end), "InputFormat", "yyyy-MM-dd HH:mm:SS");
        
        dates = [dates; epoch + seconds(time)];
    end 
  
end 

function [px, py] = euler_step(px, py, current_time, time_step, uu, vv, lon, lat, dates)
    n_particles = numel(px);
    %-----------------
    % Find current speed
    time_idx = get_index(current_time, dates);
    time_idx = repelem(time_idx, n_particles, 1);
    
    lon_idx = get_index(px, lon);
    lat_idx = get_index(py, lat);  
    
    pu = bilinear_interp(lon_idx, lat_idx, time_idx, uu);
    pv = bilinear_interp(lon_idx, lat_idx, time_idx, vv);
    
    nans = isnan(pu) | isnan(pv);    
    [pudeg, pvdeg] = m2deg(pu, pv, py); 
    
    %-----------------
    % Diffusion velocity
    du = (2*rand(n_particles, 1) - 1.) * 1.e-5 * sqrt(2.0 * time_step) / time_step;
    dv = (2*rand(n_particles, 1) - 1.) * 1.e-5 * sqrt(2.0 * time_step) / time_step;
    
    %-----------------
    % Update positions
    px(~nans) = px(~nans) + (pudeg(~nans) + du(~nans)) * time_step;
    py(~nans) = py(~nans) + (pvdeg(~nans) + dv(~nans)) * time_step;
    
end

function [px, py] = rk2_step(px, py, current_time, time_step, uu, vv, lon, lat, dates)
    n_particles = numel(px);
    %-----------------
    % Predictor step
    time_idx1 = get_index(current_time, dates);
    time_idx1 = repelem(time_idx1, n_particles, 1);
    lon_idx1 = get_index(px, lon);
    lat_idx1 = get_index(py, lat);  
    
    pu1 = bilinear_interp(lon_idx1, lat_idx1, time_idx1, uu);
    pv1 = bilinear_interp(lon_idx1, lat_idx1, time_idx1, vv);
    
    nans = isnan(pu1) | isnan(pv1);    
    [pudeg1, pvdeg1] = m2deg(pu1, pv1, py); 
    
    du1 = (2*rand(n_particles, 1) - 1.) * 1.e-5 * sqrt(2.0 * time_step) / time_step;
    dv1 = (2*rand(n_particles, 1) - 1.) * 1.e-5 * sqrt(2.0 * time_step) / time_step;
    
    x_pred = px;
    y_pred = py;
    dx_pred = zeros(size(px));
    dy_pred = zeros(size(py));
    
    dx_pred(~nans) = (pudeg1(~nans) + du1(~nans));
    dy_pred(~nans) = (pvdeg1(~nans) + dv1(~nans));  
    
    x_pred = x_pred + dx_pred * time_step;
    y_pred = y_pred + dy_pred * time_step;  
    
    %-----------------
    % Corrector step
    half_time = current_time + seconds(0.5*time_step);
    time_idx2 = get_index(half_time, dates);
    time_idx2 = repelem(time_idx2, n_particles, 1);
    lon_idx2 = get_index(x_pred, lon);
    lat_idx2 = get_index(y_pred, lat);  
    
    pu2 = bilinear_interp(lon_idx2, lat_idx2, time_idx2, uu);
    pv2 = bilinear_interp(lon_idx2, lat_idx2, time_idx2, vv);
    
    nans = isnan(pu2) | isnan(pv2);
    [pudeg2, pvdeg2] = m2deg(pu2, pv2, y_pred);
    
    du2 = (2*rand(n_particles, 1) - 1.) * 1.e-5 * sqrt(2.0 * time_step) / time_step;
    dv2 = (2*rand(n_particles, 1) - 1.) * 1.e-5 * sqrt(2.0 * time_step) / time_step;
    
    dx_corr = zeros(size(px));
    dy_corr = zeros(size(py));
    
    dx_corr(~nans) = (pudeg2(~nans) + du2(~nans));
    dy_corr(~nans) = (pvdeg2(~nans) + dv2(~nans));  
    
    %-----------------
    % Update positions
    px = px + (dx_pred + dx_corr) * (0.5*time_step);
    py = py + (dy_pred + dy_corr) * (0.5*time_step);
    
end 

function [c] = bilinear_interp(x, y, t, field)
    x0 = floor(x);
    x1 = ceil(x);
    y0 = floor(y);
    y1 = ceil(y);
    
    t0 = floor(t);
    t1 = ceil(t);
    
    xd = zeros(size(x));
    where = x1 ~= x0;
    xd(where) = (x(where) - x0(where))./(x1(where) - x0(where));
    
    yd = zeros(size(y));
    where = y1 ~= y0;
    yd(where) = (y(where) - y0(where))./(y1(where) - y0(where));
    
    td = zeros(size(t));
    where = t1 ~= t0;
    td(where) = (t(where) - t0(where))./(t1(where) - t0(where));
    
    lin_idx000 = sub2ind(size(field), x0, y0, t0);
    lin_idx100 = sub2ind(size(field), x1, y0, t0);
    lin_idx010 = sub2ind(size(field), x0, y1, t0);
    lin_idx110 = sub2ind(size(field), x1, y1, t0);
    lin_idx001 = sub2ind(size(field), x0, y0, t1);
    lin_idx101 = sub2ind(size(field), x1, y0, t1);
    lin_idx011 = sub2ind(size(field), x0, y1, t1);
    lin_idx111 = sub2ind(size(field), x1, y1, t1);
    
    c00 = field(lin_idx000).*(1-xd) + field(lin_idx100).*xd;
    c10 = field(lin_idx010).*(1-xd) + field(lin_idx110).*xd;
    c01 = field(lin_idx001).*(1-xd) + field(lin_idx101).*xd;
    c11 = field(lin_idx011).*(1-xd) + field(lin_idx111).*xd;
    
    c0 = c00.*(1 - yd) + c10.*yd;
    c1 = c01.*(1 - yd) + c11.*yd;
    
    c = c0.*(1-td) + c1.*td;
    
end 

function [r_lon, r_lat] = m2deg(xm, ym, lat)
    radius_earth = 6371e3;
    r_lon = xm ./ (radius_earth * pi / 180.0) ./ cos(pi * lat / 180.0);
    r_lat = ym ./ (radius_earth * pi / 180.0);
end

function idx = get_index(x, values)
    idx = interp1(values, 1:numel(values), x, 'linear', 'extrap');       
end 






